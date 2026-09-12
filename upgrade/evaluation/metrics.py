"""
Height-estimation metrics for the P0 baseline harness.

Pure NumPy, no project dependencies, so it stays independently testable
(see tests/test_metrics.py). All functions ignore invalid / nodata pixels
using the same masking convention as backend/app/services/validation.py.

Two families of metric live here, and the distinction is the whole point of P0:

  * Direct metric error (mae, rmse, delta_thresholds, tall_structure_mae):
    "how many meters off is the output as-is". Exposes the calibration gap.
  * Scale-invariant error (si_rmse, affine_aligned_rmse): "how good is the
    predicted *shape*, ignoring global scale/shift". Fair to a relative model.

boundary_f_score measures edge sharpness (the crisp-building-wall quality).
"""
from __future__ import annotations

import numpy as np

_NODATA_LO = -9000.0
_NODATA_HI = 15000.0


def valid_mask(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels usable for comparison (finite, in physical range)."""
    return (
        np.isfinite(pred)
        & np.isfinite(ref)
        & (ref > _NODATA_LO) & (ref < _NODATA_HI)
        & (pred > _NODATA_LO) & (pred < _NODATA_HI)
    )


def _masked_pair(pred: np.ndarray, ref: np.ndarray):
    m = valid_mask(pred, ref)
    return pred[m].astype(np.float64), ref[m].astype(np.float64)


def mae(pred: np.ndarray, ref: np.ndarray) -> float:
    p, r = _masked_pair(pred, ref)
    if p.size == 0:
        return float("nan")
    return float(np.mean(np.abs(p - r)))


def rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    p, r = _masked_pair(pred, ref)
    if p.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((p - r) ** 2)))


def delta_thresholds(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
    """Return (delta1, delta2, delta3): fraction of pixels with
    max(pred/ref, ref/pred) < 1.25**k, for k = 1, 2, 3."""
    p, r = _masked_pair(pred, ref)
    if p.size == 0:
        return (float("nan"),) * 3

    if np.all(p > 0.01) and np.all(r > 0.01):
        ps, rs = p, r
    else:
        # Shift both so the ratio is well-defined for zero/negative elevations.
        shift = max(0.0, -float(min(p.min(), r.min()))) + 1.0
        ps, rs = p + shift, r + shift

    ratio = np.maximum(ps / rs, rs / ps)
    d1 = float(np.mean(ratio < 1.25))
    d2 = float(np.mean(ratio < 1.25 ** 2))
    d3 = float(np.mean(ratio < 1.25 ** 3))
    return d1, d2, d3


def si_rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """Scale-invariant RMSE in log space: sqrt(mean(d^2) - mean(d)^2),
    where d = log(pred) - log(ref). Zero under a pure global scale."""
    m = valid_mask(pred, ref) & (pred > 1e-6) & (ref > 1e-6)
    p = pred[m].astype(np.float64)
    r = ref[m].astype(np.float64)
    if p.size < 2:
        return float("nan")
    d = np.log(p) - np.log(r)
    return float(np.sqrt(max(np.mean(d ** 2) - np.mean(d) ** 2, 0.0)))


def affine_fit(pred: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    """Least-squares scale s and shift t minimizing ||s*pred + t - ref|| over
    valid pixels. Applying (s, t) maps a relative prediction onto metric truth,
    removing the unknown global scale/offset the model was never trained on."""
    p, r = _masked_pair(pred, ref)
    if p.size < 2:
        return 1.0, 0.0
    var_p = np.var(p)
    if var_p < 1e-12:
        return 1.0, float(np.mean(r) - np.mean(p))
    s = float(np.cov(p, r, bias=True)[0, 1] / var_p)
    t = float(np.mean(r) - s * np.mean(p))
    return s, t


def affine_aligned_rmse(pred: np.ndarray, ref: np.ndarray) -> float:
    """RMSE after best-fit scale+shift alignment (see affine_fit). The honest
    way to score a relative-depth model's shape against metric truth."""
    p, r = _masked_pair(pred, ref)
    if p.size < 2:
        return float("nan")
    s, t = affine_fit(pred, ref)
    aligned = s * p + t
    return float(np.sqrt(np.mean((aligned - r) ** 2)))


def _edge_map(arr: np.ndarray, edge_threshold: float) -> np.ndarray:
    """Binary discontinuity map: gradient magnitude above a threshold."""
    a = np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    gy, gx = np.gradient(a)
    mag = np.hypot(gx, gy)
    return mag > edge_threshold


def boundary_f_score(pred: np.ndarray, ref: np.ndarray, edge_threshold: float,
                     tol: int = 2) -> float:
    """F1 between predicted and reference depth-discontinuity maps, with a `tol`-pixel
    matching tolerance (BSDS-style). A predicted edge counts if it's within `tol`
    pixels of a reference edge (and vice versa). Without tolerance a 1-2 px offset —
    normal for a neural edge vs a LiDAR raster — collapses the score to 0, which is
    why exact-match Boundary-F read near-zero on the P0 baseline. tol=0 = exact."""
    pe = _edge_map(pred, edge_threshold)
    re = _edge_map(ref, edge_threshold)
    p_sum, r_sum = int(pe.sum()), int(re.sum())
    if p_sum == 0 and r_sum == 0:
        return 1.0
    if p_sum == 0 or r_sum == 0:
        return 0.0
    if tol > 0:
        from scipy.ndimage import binary_dilation
        st = np.ones((2 * tol + 1, 2 * tol + 1), dtype=bool)
        pe_d, re_d = binary_dilation(pe, st), binary_dilation(re, st)
    else:
        pe_d, re_d = pe, re
    tp_p = int(np.sum(pe & re_d))   # predicted edges near a ref edge -> precision
    tp_r = int(np.sum(re & pe_d))   # ref edges near a predicted edge -> recall
    if tp_p == 0 or tp_r == 0:
        return 0.0
    precision = tp_p / p_sum
    recall = tp_r / r_sum
    return float(2 * precision * recall / (precision + recall))


def tall_structure_mae(pred: np.ndarray, ref: np.ndarray, height_threshold: float = 15.0) -> float:
    """MAE computed only over pixels where the ground truth exceeds
    height_threshold meters. Surfaces the documented tall-building
    underestimation that ground-dominated averages hide."""
    m = valid_mask(pred, ref) & (ref > height_threshold)
    if not np.any(m):
        return float("nan")
    return float(np.mean(np.abs(pred[m].astype(np.float64) - ref[m].astype(np.float64))))
