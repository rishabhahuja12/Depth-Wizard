"""
Tests for MixedMetricDataset — blending multiple harmonized height sources.

Uses in-memory fake sources (no rasterio / no downloads) so the mixing, balanced
sampling, GSD-resampling and meters-conforming are all exercised end to end.

Run:
    .venv/Scripts/python.exe upgrade/training/tests/test_multi_dataset.py
"""
import sys
from pathlib import Path

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAIN_DIR))

import multi_dataset as mds  # noqa: E402


class FakeSource:
    """Minimal height source: N tiles of a fixed RGB + height (meters)."""
    def __init__(self, name, gsd, n, size=64, height_val=12.0):
        self.name, self.gsd, self.n = name, gsd, n
        self._rgb = (np.ones((size, size, 3)) * 128).astype(np.uint8)
        self._h = (np.ones((size, size)) * height_val).astype(np.float32)

    def __len__(self):
        return self.n

    def raw(self, i):
        return self._rgb, self._h


def test_balanced_length_follows_weights_not_sizes():
    big = FakeSource("gamus", 0.33, 1000)
    small = FakeSource("canopy", 1.5, 10)
    ds = mds.MixedMetricDataset([big, small], weights={"gamus": 1.0, "canopy": 1.0},
                                target_gsd=0.5, crop=32, total_per_epoch=100)
    # equal weights -> ~half from each despite the 100x size gap
    counts = ds.source_counts()
    assert counts["gamus"] == 50 and counts["canopy"] == 50
    assert len(ds) == 100


def test_getitem_returns_crop_sized_meters_tensor():
    src = FakeSource("gamus", 0.33, 4, size=64, height_val=20.0)
    ds = mds.MixedMetricDataset([src], weights={"gamus": 1.0},
                                target_gsd=0.33, crop=48, total_per_epoch=4)
    rgb_t, depth_t = ds[0]
    assert tuple(rgb_t.shape) == (3, 48, 48), f"rgb shape {tuple(rgb_t.shape)}"
    assert tuple(depth_t.shape) == (48, 48), f"depth shape {tuple(depth_t.shape)}"
    # height must survive in METERS (~20), not be normalized
    assert abs(float(depth_t.max()) - 20.0) < 1e-3, f"height not preserved: {depth_t.max()}"


def test_resampling_changes_pixel_grid():
    # A coarse 1.5 m source read onto a 0.5 m grid triples the tile before cropping,
    # so a crop larger than the source's native size is still filled.
    src = FakeSource("canopy", 1.5, 2, size=32, height_val=8.0)
    ds = mds.MixedMetricDataset([src], weights={"canopy": 1.0},
                                target_gsd=0.5, crop=64, total_per_epoch=2)
    rgb_t, depth_t = ds[0]
    assert tuple(depth_t.shape) == (64, 64)
    assert abs(float(depth_t.max()) - 8.0) < 1e-3


def test_cell_crop_preserves_scale_not_resize():
    # G3: a tile larger than crop must be CROPPED to a cell at target_gsd, not
    # squashed to crop (which silently undoes the GSD harmonization).
    class Ramp:
        name, gsd = "ramp", 0.5
        def __len__(self):
            return 1
        def raw(self, i):
            rows = (np.arange(128, dtype=np.float32)[:, None] * np.ones((1, 128), np.float32))
            rgb = np.repeat(rows[..., None], 3, axis=-1).astype(np.uint8)
            return rgb, rows                      # height = row index 0..127

    ds = mds.MixedMetricDataset([Ramp()], weights={"ramp": 1.0},
                                target_gsd=0.5, crop=32, total_per_epoch=1)  # factor 1: 128 -> 32 cell
    _, depth_t = ds[0]
    assert tuple(depth_t.shape) == (32, 32)
    # center cell = rows ~48..79 -> mid-range values. A resize would keep the full
    # 0..127 span, so this range check fails if the tile were squashed.
    assert 40 < float(depth_t.min()) and float(depth_t.max()) < 100, \
        f"tile resized (full span) not cell-cropped: [{float(depth_t.min())}, {float(depth_t.max())}]"


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
