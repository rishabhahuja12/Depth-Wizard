"""
Tests for Open-Canopy / GBH source configs — exercised against REAL (tiny)
GeoTIFFs written with rasterio, so the paired-raster reading is genuinely
validated, not mocked.

Real-world caveat: the exact folder names differ per dataset and can only be
confirmed against the actual download; the readers take those as parameters.

Run:
    .venv/Scripts/python.exe upgrade/training/tests/test_aux_datasets.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAIN_DIR))

import aux_datasets as aux  # noqa: E402


def _write_tif(path: Path, arr: np.ndarray, count: int):
    import rasterio
    from rasterio.transform import from_origin
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        arr = arr[None, ...]
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                       count=count, dtype=arr.dtype, transform=from_origin(0, 0, 1, 1)) as ds:
        for b in range(count):
            ds.write(arr[b], b + 1)


def _fake_paired_dataset(root: Path, rgb_sub: str, h_sub: str, n=2, size=40, h_val=9.0):
    for i in range(n):
        rgb = (np.ones((3, size, size)) * 100).astype(np.uint8)
        h = (np.ones((size, size)) * h_val).astype(np.float32)
        _write_tif(root / rgb_sub / f"tile_{i}.tif", rgb, 3)
        _write_tif(root / h_sub / f"tile_{i}.tif", h, 1)


def test_open_canopy_source_pairs_and_reads_meters():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _fake_paired_dataset(root, "images", "canopy_height", n=3, h_val=14.0)
        src = aux.open_canopy_source(root, rgb_subdir="images", height_subdir="canopy_height")
        assert src.name == "open_canopy" and abs(src.gsd - 1.5) < 1e-6
        assert len(src) == 3
        rgb, height = src.raw(0)
        assert rgb.shape[-1] == 3
        assert abs(float(height.max()) - 14.0) < 1e-3, "canopy height not read in meters"


def test_gbh_source_defaults_gsd_3m():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _fake_paired_dataset(root, "rgb", "ndsm", n=2, h_val=22.0)
        src = aux.gbh_source(root, rgb_subdir="rgb", height_subdir="ndsm")
        assert src.name == "gbh" and abs(src.gsd - 3.0) < 1e-6
        assert len(src) == 2
        _, height = src.raw(1)
        assert abs(float(height.max()) - 22.0) < 1e-3


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
        else:
            passed += 1
            print(f"ok    {t.__name__}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
