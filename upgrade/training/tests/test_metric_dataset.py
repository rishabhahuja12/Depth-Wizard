"""
Known-answer tests for the pure logic in metric_dataset.

The headline guard is test_prepare_target_preserves_meters: the P1 dataset must
NOT normalize the target (that bug is exactly what P1 removes).

Run:
    .venv/Scripts/python.exe upgrade/training/tests/test_metric_dataset.py
"""
import sys
from pathlib import Path

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAIN_DIR))

import metric_dataset as md  # noqa: E402


def test_prepare_target_preserves_meters():
    # A patch with real heights up to 42.5 m must come back with max ~42.5,
    # NOT 1.0 — proving no [0,1] normalization sneaks in.
    agl = np.array([[0.0, 10.0], [25.0, 42.5]], dtype=np.float32)
    out = md.prepare_target(agl)
    assert abs(float(out.max()) - 42.5) < 1e-4, f"max not preserved: {out.max()}"
    assert abs(float(out.min()) - 0.0) < 1e-4


def test_prepare_target_cleans_nan_and_negatives():
    agl = np.array([[np.nan, -5.0], [3.0, 10.0]], dtype=np.float32)
    out = md.prepare_target(agl)
    assert np.isfinite(out).all(), "NaN survived"
    assert float(out.min()) >= 0.0, "negative height survived"
    assert abs(float(out[1, 1]) - 10.0) < 1e-4, "valid height altered"


def test_random_crop_box_is_within_bounds_and_sized():
    rng = np.random.default_rng(0)
    for _ in range(20):
        r1, r2, c1, c2 = md.random_crop_box(1024, 1024, 518, rng)
        assert 0 <= r1 and r2 <= 1024 and 0 <= c1 and c2 <= 1024
        assert (r2 - r1) == 518 and (c2 - c1) == 518


def test_random_crop_box_handles_crop_larger_than_image():
    r1, r2, c1, c2 = md.random_crop_box(300, 300, 518, np.random.default_rng(0))
    assert (r1, r2, c1, c2) == (0, 300, 0, 300)  # clamp to full image


def test_apply_augment_hflip_transforms_both_identically():
    rgb = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rgb_a, depth_a = md.apply_augment(rgb, depth, hflip=True, vflip=False, k_rot=0)
    assert np.array_equal(rgb_a, np.fliplr(rgb))
    assert np.array_equal(depth_a, np.fliplr(depth))


def test_apply_augment_rot90_transforms_both_identically():
    rgb = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rgb_a, depth_a = md.apply_augment(rgb, depth, hflip=False, vflip=False, k_rot=1)
    assert np.array_equal(rgb_a, np.rot90(rgb, k=1))
    assert np.array_equal(depth_a, np.rot90(depth, k=1))


def test_apply_augment_identity_when_no_ops():
    rgb = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    depth = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rgb_a, depth_a = md.apply_augment(rgb, depth, hflip=False, vflip=False, k_rot=0)
    assert np.array_equal(rgb_a, rgb) and np.array_equal(depth_a, depth)


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
