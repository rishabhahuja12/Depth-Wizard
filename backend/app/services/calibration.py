"""
Two-Component Elevation Decomposition.
DSM(x,y) = DTM_base(x,y) + α · nDSM_pred(x,y)

For MVP: uses statistical prior (no real SRTM download).
"""
import numpy as np
from dataclasses import dataclass
from app.logging_config import log


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
) -> CalibrationResult:
    """
    Convert relative depth [0,1] to calibrated DSM.

    For georeferenced: scale to approximate metric range using GSD and statistical prior.
    For non-georeferenced: keep as relative [0,1].
    """
    if is_georef and gsd is not None:
        return _metric_calibration(relative_depth, gsd, target_range)
    else:
        return _relative_calibration(relative_depth)


def _metric_calibration(depth: np.ndarray, gsd: float, target_range: float) -> CalibrationResult:
    """
    Statistical prior calibration.
    Assumes urban scene with buildings up to target_range meters.
    α = target_range / depth_range
    """
    # Ground level = 5th percentile
    ground = np.percentile(depth, 5)
    ndsm = np.maximum(depth - ground, 0.0)

    # Scale factor from statistical prior
    depth_range = np.percentile(ndsm, 99) - np.percentile(ndsm, 1)
    if depth_range < 1e-6:
        alpha = 1.0
    else:
        alpha = target_range / depth_range

    # Base elevation (simulated DTM at 0m for MVP)
    dtm_base = 0.0
    dsm = dtm_base + alpha * ndsm

    log.info("Metric calibration", alpha=f"{alpha:.2f}",
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
