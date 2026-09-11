"""
P2 — tile-based high-resolution inference (PatchFusion/PRO-style).

The detail we want comes from running the model at native resolution via
overlapping patches, consistency-blended so grid seams vanish — NOT from
super-resolving/inventing pixels. This is the EXPORT path (~9 ViT-L passes ≈
3.5–5 s); the interactive path stays a single-pass Live Mode (~350 ms).

The model forward is injected (predict_fn), so the tiling/stitching math here is
pure and unit-tested; the real ViT-L pass is supplied by the caller.
"""
from __future__ import annotations

import numpy as np


def tile_positions(H: int, W: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """Top-left (row, col) of each patch. Step = tile - overlap. The last row/col
    is snapped to the edge so coverage is complete."""
    step = max(1, tile - overlap)

    def axis(n: int) -> list[int]:
        if tile >= n:
            return [0]
        pos = list(range(0, n - tile + 1, step))
        if pos[-1] != n - tile:
            pos.append(n - tile)
        return pos

    return [(r, c) for r in axis(H) for c in axis(W)]


def blend_window(tile: int) -> np.ndarray:
    """2D partition-of-unity window (separable Hann-like), peaking at the centre
    and tapering to a small positive value at the edges, so overlapping patches
    blend without seams. Never exactly 0 (keeps the accumulator well-posed)."""
    x = np.linspace(-np.pi, np.pi, tile)
    w1 = 0.5 * (1.0 + np.cos(x))          # 1 at centre -> 0 at ends
    w1 = w1 * 0.98 + 0.02                 # floor so edges still contribute
    return np.outer(w1, w1).astype(np.float32)


def _stitch(patches, positions, out_hw, tile):
    H, W = out_hw
    acc = np.zeros((H, W), dtype=np.float64)
    wsum = np.zeros((H, W), dtype=np.float64)
    win_full = blend_window(tile)
    for (r0, c0), patch in zip(positions, patches):
        h, w = patch.shape[:2]
        win = win_full[:h, :w] if (h, w) != win_full.shape else win_full
        acc[r0:r0 + h, c0:c0 + w] += patch.astype(np.float64) * win
        wsum[r0:r0 + h, c0:c0 + w] += win
    return (acc / np.maximum(wsum, 1e-8)).astype(np.float32)


def tiled_predict(predict_fn, image: np.ndarray, tile: int, overlap: int) -> np.ndarray:
    """Run predict_fn on each overlapping patch of `image` (H,W,C) and blend.
    predict_fn(patch)->(h,w) depth. Returns (H,W)."""
    H, W = image.shape[:2]
    positions = tile_positions(H, W, tile, overlap)
    patches = [predict_fn(image[r:r + tile, c:c + tile]) for r, c in positions]
    return _stitch(patches, positions, (H, W), tile)


def tiled_predict_positioned(predict_fn_at, image_shape, tile: int, overlap: int) -> np.ndarray:
    """Variant where the predictor is built per-position: predict_fn_at(r0,c0)->
    fn(patch). Used to test exact reconstruction; mirrors tiled_predict otherwise."""
    H, W = image_shape[:2]
    positions = tile_positions(H, W, tile, overlap)
    dummy = np.zeros(image_shape, dtype=np.uint8)
    patches = [predict_fn_at(r, c)(dummy[r:r + tile, c:c + tile]) for r, c in positions]
    return _stitch(patches, positions, (H, W), tile)
