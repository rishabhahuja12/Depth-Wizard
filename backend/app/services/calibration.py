"""
Two-Component Elevation Decomposition.
DSM(x,y) = DTM_base(x,y) + α · nDSM_pred(x,y)

For MVP: uses statistical prior (no real SRTM download).
"""
import numpy as np
from dataclasses import dataclass
from app.logging_config import log


from scipy.ndimage import minimum_filter, gaussian_filter


@dataclass
class CalibrationResult:
    dsm: np.ndarray           # (H, W) float32, calibrated elevation
    alpha: float               # scale factor
    dsm_min: float
    dsm_max: float
    dsm_mean: float
    mode: str                  # "metric_prior" or "relative"
    unit: str                  # "meters" or "relative"


def calibrate_depth(
    relative_depth: np.ndarray,
    is_georef: bool,
    gsd: float = None,
    target_range: float = 50.0,  # default assumed max building height in meters
    is_metric: bool = False,     # model already outputs meters-above-ground (fine-tuned)
) -> CalibrationResult:
    """
    Convert a model's depth prediction to a calibrated DSM.

    - is_metric=True: the model was metric fine-tuned and ALREADY outputs
      meters-above-ground (nDSM). Pass it through as meters — do NOT apply the
      statistical alpha, or we'd double-scale a value that's already correct.
    - is_metric=False (relative model): georef -> scale to an approximate metric
      range via the GSD/statistical prior; non-georef -> keep relative [0,1].
    """
    if is_metric:
        return _metric_model_passthrough(relative_depth)
    if is_georef and gsd is not None:
        return _metric_calibration(relative_depth, gsd, target_range)
    else:
        return _relative_calibration(relative_depth)


def _metric_model_passthrough(depth_meters: np.ndarray) -> CalibrationResult:
    """Trust a metric fine-tuned model's output: it is already meters-above-ground.

    Clean it (NaN/inf/neg -> 0) and return as-is. The absolute DTM base (real
    ground elevation, e.g. from SRTM) is added elsewhere; here nDSM stands alone.
    """
    ndsm = np.maximum(np.nan_to_num(depth_meters.astype(np.float32),
                                    nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    log.info("Metric model passthrough (no re-scaling)",
             dsm_range=f"[{ndsm.min():.1f}, {ndsm.max():.1f}]m")
    return CalibrationResult(
        dsm=ndsm, alpha=1.0,
        dsm_min=float(ndsm.min()), dsm_max=float(ndsm.max()), dsm_mean=float(ndsm.mean()),
        mode="metric_model", unit="meters",
    )


def _metric_calibration(depth: np.ndarray, gsd: float, target_range: float) -> CalibrationResult:
    """
    Two-Component Elevation Decomposition:
    DSM(x, y) = DTM_base(x, y) + α · nDSM(x, y)

    - DTM_base is modeled using spatial morphological filtering to extract bare-earth topography.
    - GSD scales the effective relief range according to sensor resolution and ground footprint.
    - α maps normalized structural variations to metric elevation (meters).
    """
    safe_gsd = max(float(gsd), 0.05) if gsd is not None else 1.0

    # 1. Incorporate GSD into effective height range
    # VHR imagery (~0.3-0.5m) resolves single buildings (target_range ~ 30-50m).
    # Medium resolution (~5-10m) captures wider spatial extents with larger topographic relief.
    gsd_scale = float(np.clip(np.sqrt(safe_gsd / 0.5), 0.7, 3.0))
    effective_range = target_range * gsd_scale

    # 2. Bare-Earth DTM baseline modeling via morphological minimum + gaussian smoothing
    # Kernel size corresponds to typical maximum structural footprint (~30-50m) in pixels
    min_dim = min(depth.shape)
    if min_dim >= 12:
        max_kernel = max(3, min_dim // 6)
        kernel_px = int(np.clip(40.0 / safe_gsd, 3, max_kernel))
        if kernel_px % 2 == 0:
            kernel_px += 1
        dtm_raw = minimum_filter(depth, size=kernel_px)
        dtm_base = gaussian_filter(dtm_raw, sigma=max(kernel_px / 3.0, 1.0))
    else:
        # Fallback for very small patches
        dtm_base = np.full_like(depth, np.percentile(depth, 5))

    # 3. Structural height above ground (nDSM)
    ndsm = np.maximum(depth - dtm_base, 0.0)

    # 4. Scale factor α computed from statistical prior without percentile clipping redundancy
    ndsm_span = float(np.percentile(ndsm, 99) - np.percentile(ndsm, 5))
    if ndsm_span < 1e-6:
        depth_span = float(np.percentile(depth, 99) - np.percentile(depth, 1))
        alpha = (effective_range / depth_span) if depth_span > 1e-6 else 1.0
    else:
        alpha = effective_range / ndsm_span

    # 5. Composite DSM: bare-earth terrain relief + structural elevation
    dtm_metric = (dtm_base - float(dtm_base.min())) * (alpha * 0.4)
    ndsm_metric = alpha * ndsm
    dsm = dtm_metric + ndsm_metric

    log.info("Metric calibration", alpha=f"{alpha:.2f}", gsd=f"{safe_gsd:.2f}m",
             dsm_range=f"[{dsm.min():.1f}, {dsm.max():.1f}]m")

    return CalibrationResult(
        dsm=dsm.astype(np.float32),
        alpha=float(alpha),
        dsm_min=float(dsm.min()),
        dsm_max=float(dsm.max()),
        dsm_mean=float(dsm.mean()),
        mode="metric_prior",
        unit="meters"
    )


def _relative_calibration(depth: np.ndarray) -> CalibrationResult:
    """Keep as relative depth normalized to [0, 1]."""
    log.info("Relative calibration (non-georeferenced)")
    return CalibrationResult(
        dsm=depth.astype(np.float32),
        alpha=1.0,
        dsm_min=float(depth.min()),
        dsm_max=float(depth.max()),
        dsm_mean=float(depth.mean()),
        mode="relative",
        unit="relative"
    )
