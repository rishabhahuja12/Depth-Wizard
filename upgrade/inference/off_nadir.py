"""
P4 — off-nadir DETECT + FLAG (never correct, never synthesize; research §3.3).

Two hard constraints kill any real fix: no RPC metadata (can't de-lean), and we
cannot synthesize oblique (RGB, nDSM) pairs from nadir data without corrupting
labels. So DepthWizard is a nadir instrument that DETECTS likely oblique inputs
and shows an honest warning, degrading gracefully — it does not pretend to correct
the geometry.

The cue is metadata-free edge-orientation anisotropy: near-nadir aerial imagery
has fairly isotropic edge directions; strong oblique/tilt tends to introduce a
dominant lean direction, raising anisotropy. This is a COARSE flag, not a
measurement of the off-nadir angle — stated as a known limitation, not a feature.
"""
from __future__ import annotations

import numpy as np


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    return img.astype(np.float32)


def obliqueness_score(img: np.ndarray) -> float:
    """Edge-orientation anisotropy in [0, 1]. 0 = isotropic (nadir-like), higher =
    a dominant edge direction (possible oblique lean). Uses the structure-tensor
    coherence, a standard, cheap orientation-anisotropy measure."""
    g = _gray(img)
    gy, gx = np.gradient(g)
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    trace = jxx + jyy
    if trace < 1e-8:
        return 0.0
    # coherence = (λ1-λ2)/(λ1+λ2) of the structure tensor, in [0,1]
    diff = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
    return float(np.clip(diff / trace, 0.0, 1.0))


def is_oblique(score: float, threshold: float = 0.5) -> bool:
    return bool(score >= threshold)


def oblique_flag(score: float, threshold: float = 0.5) -> dict:
    """Structured flag for the UI/API layer."""
    ob = is_oblique(score, threshold)
    return {
        "oblique": ob,
        "score": round(float(score), 3),
        "message": ("⚠ Oblique input suspected — reduced accuracy (nadir-only model)"
                    if ob else "Near-nadir — nominal"),
    }


def flag_image(img: np.ndarray, threshold: float = 0.5) -> dict:
    return oblique_flag(obliqueness_score(img), threshold)
