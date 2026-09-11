"""
Tests for P2 — tile-based high-res inference (Export path; Live Mode stays single-pass).

Overlapping patches predicted separately then consistency-blended so seams vanish.
The model forward is injected (predict_fn) so the stitching math is tested with no
GPU. Run:
    .venv/Scripts/python.exe upgrade/inference/tests/test_tiling.py
"""
import sys
from pathlib import Path

import numpy as np

INF_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(INF_DIR))

import tiling  # noqa: E402


def test_tile_positions_cover_and_overlap():
    pos = tiling.tile_positions(1024, 1024, tile=512, overlap=128)
    assert (0, 0) in pos
    # last tile must reach the bottom-right edge
    assert any(r + 512 == 1024 and c + 512 == 1024 for r, c in pos)
    # step = tile - overlap = 384
    rows = sorted({r for r, _ in pos})
    assert rows[1] - rows[0] == 384


def test_blend_window_peaks_at_center():
    w = tiling.blend_window(64)
    assert w.shape == (64, 64)
    assert w[32, 32] > w[0, 0]           # center weighted more than the edge
    assert w.min() >= 0.0


def test_stitch_constant_field_is_seamless():
    # Every tile predicts the constant 7.0 -> stitched result is 7.0 everywhere,
    # with no seam artifacts at the overlaps.
    H = W = 256
    def predict_fn(patch):
        return np.full(patch.shape[:2], 7.0, dtype=np.float32)
    img = np.zeros((H, W, 3), dtype=np.uint8)
    out = tiling.tiled_predict(predict_fn, img, tile=128, overlap=32)
    assert out.shape == (H, W)
    assert np.allclose(out, 7.0, atol=1e-4), f"seam artifact: range [{out.min()},{out.max()}]"


def test_stitch_recovers_a_gradient():
    # A predictor that returns the true per-pixel value must be reconstructed
    # exactly by the weighted blend (partition-of-unity property).
    H = W = 200
    truth = np.tile(np.linspace(0, 50, W, dtype=np.float32), (H, 1))
    def predict_fn_at(r0, c0):
        def f(patch):
            h, w = patch.shape[:2]
            return truth[r0:r0 + h, c0:c0 + w].copy()
        return f
    out = tiling.tiled_predict_positioned(predict_fn_at, (H, W, 3), tile=80, overlap=24)
    assert np.allclose(out, truth, atol=1e-3), f"max err {np.abs(out-truth).max()}"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
        else:
            passed += 1
            print(f"ok    {t.__name__}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
