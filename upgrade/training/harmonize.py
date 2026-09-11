"""
Multi-dataset harmonization (research §2.9).

Rule: one backbone, ONE height head, fed by all *dense height-in-meters* sources.
Sources differ only in "plumbing" — GSD, radiometry, value range — which we fix
here so GAMUS (0.33 m) / Open-Canopy (1.5 m) / GBH (~3 m) become one shared target:

  * resample every source to a common GSD (or, later, condition the model on GSD),
  * force every label to meters-above-ground (nDSM / canopy height are already this),
  * balance batches so a large set doesn't drown a small forested/global slice.

NOT harmonizable: label *semantics*. Footprint-only data (SpaceNet-4) is a
different task and never joins this height loss — see research §2.9 / OFF_NADIR.

Pure functions only (unit-tested); the raster I/O lives in the source adapters.
"""
from __future__ import annotations

import numpy as np


def resample_factor(src_gsd: float, dst_gsd: float) -> float:
    """Pixels-per-pixel scale to move a src-GSD raster onto a dst-GSD grid.
    Coarser source (larger GSD) -> factor > 1 (upsampled to the finer grid)."""
    if dst_gsd <= 0:
        raise ValueError("dst_gsd must be > 0")
    return src_gsd / dst_gsd


def resampled_size(n: int, src_gsd: float, dst_gsd: float) -> int:
    return int(round(n * resample_factor(src_gsd, dst_gsd)))


def resample_to_gsd(arr: np.ndarray, src_gsd: float, dst_gsd: float, order: int = 1) -> np.ndarray:
    """Resample a 2D (or HxWxC) array from src_gsd to dst_gsd. order=1 bilinear
    for heights/RGB; order=0 nearest for label masks."""
    if abs(src_gsd - dst_gsd) < 1e-9:
        return arr
    from scipy.ndimage import zoom
    factor = resample_factor(src_gsd, dst_gsd)
    if arr.ndim == 3:
        return zoom(arr, (factor, factor, 1.0), order=order)
    return zoom(arr, factor, order=order)


def conform_meters_above_ground(height: np.ndarray) -> np.ndarray:
    """Force a height raster to clean meters-above-ground: NaN/inf -> 0, negatives
    (water, LiDAR noise) -> 0 (ground). nDSM and canopy height are already AGL."""
    out = np.nan_to_num(height.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(out, 0.0)


def balanced_source_plan(sizes: dict[str, int], weights: dict[str, float],
                         total: int) -> dict[str, int]:
    """How many samples to draw per source for one balanced 'epoch' of `total`
    samples. Counts follow the weights (NOT the raw sizes), so a 200-tile forest
    slice isn't drowned by a 10k-tile urban set. Weights default to 1 each."""
    names = list(sizes.keys())
    w = np.array([max(0.0, weights.get(n, 1.0)) for n in names], dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones(len(names))
    frac = w / w.sum()
    counts = np.floor(frac * total).astype(int)
    # distribute rounding remainder to the largest-weight sources
    remainder = total - int(counts.sum())
    for i in np.argsort(-frac)[:remainder]:
        counts[i] += 1
    return {n: int(c) for n, c in zip(names, counts)}
