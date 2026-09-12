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


def test_geotiff_source_sanitizes_nodata_sentinel():
    # A height tile with a positive nodata sentinel must read as 0, not a huge value.
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rgb = (np.ones((3, 40, 40)) * 100).astype(np.uint8)
        h = np.full((40, 40), 32767.0, np.float32); h[:20] = 12.0  # half real, half sentinel
        _write_tif(root / "rgb" / "t0.tif", rgb, 3)
        _write_tif(root / "ndsm" / "t0.tif", h, 1)
        # default max_height_m=1000 -> undeclared 32767 sentinel dropped to ground
        src = aux.geonrw_source(root, rgb_subdir="rgb", height_subdir="ndsm")
        _, height = src.raw(0)
        assert abs(float(height[:20].max()) - 12.0) < 1e-3   # real heights preserved
        assert float(height[20:].max()) == 0.0               # sentinel -> ground


def test_geonrw_source_config():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _fake_paired_dataset(root, "rgb", "ndsm", n=2, h_val=18.0)
        src = aux.geonrw_source(root, rgb_subdir="rgb", height_subdir="ndsm")
        assert src.name == "geonrw" and abs(src.gsd - 1.0) < 1e-6
        assert len(src) == 2


def test_derive_ndsm_writes_above_ground():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        dsm = np.array([[110.0, 130.0], [100.0, 100.0]], np.float32)
        dtm = np.array([[100.0, 100.0], [100.0, 100.0]], np.float32)
        _write_tif(root / "dsm" / "a.tif", dsm, 1)
        _write_tif(root / "dtm" / "a.tif", dtm, 1)
        n = aux.derive_ndsm(root / "dsm", root / "dtm", root / "ndsm")
        assert n == 1
        src = aux.geonrw_source(root, rgb_subdir="dsm", height_subdir="ndsm")  # reuse reader
        _, ndsm = src.raw(0)
        assert abs(float(ndsm.max()) - 30.0) < 1e-3   # 130-100 above ground


def test_token_remap_pairs_rgb_to_agl():
    # US3D-style: JAX_004_RGB.tif <-> JAX_004_AGL.tif (different stems).
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rgb = (np.ones((3, 30, 30)) * 80).astype(np.uint8)
        h = (np.ones((30, 30)) * 7.0).astype(np.float32)
        _write_tif(root / "rgb" / "JAX_004_RGB.tif", rgb, 3)
        _write_tif(root / "agl" / "JAX_004_AGL.tif", h, 1)
        src = aux.GeoTiffHeightSource(name="us3d", gsd=0.5,
                                      rgb_dir=root / "rgb", height_dir=root / "agl",
                                      rgb_token="RGB", height_token="AGL")
        assert len(src) == 1, "token remap failed to pair RGB->AGL"
        # Without the remap it must NOT pair, and n_rgb still reports the found RGB.
        bad = aux.GeoTiffHeightSource(name="us3d", gsd=0.5,
                                      rgb_dir=root / "rgb", height_dir=root / "agl")
        assert len(bad) == 0 and bad.n_rgb == 1


def test_rgb_band_selection_picks_requested_bands():
    # 5-band tile, band b filled with value b*10; ask for bands (5,3,2).
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        arr = np.stack([np.full((20, 20), (b + 1) * 10, np.uint8) for b in range(5)])
        _write_tif(root / "rgb" / "t.tif", arr, 5)
        _write_tif(root / "h" / "t.tif", (np.ones((20, 20)) * 3.0).astype(np.float32), 1)
        src = aux.GeoTiffHeightSource(name="s", gsd=0.5, rgb_dir=root / "rgb",
                                      height_dir=root / "h", rgb_bands=(5, 3, 2))
        rgb, _ = src.raw(0)
        assert rgb[0, 0, 0] == 50 and rgb[0, 0, 1] == 30 and rgb[0, 0, 2] == 20


def test_m4heights_and_dfc2023_sources():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _fake_paired_dataset(root, "rgb", "height", n=2, h_val=15.0)
        m = aux.m4heights_source(root, rgb_subdir="rgb", height_subdir="height")
        assert m.name == "m4heights" and abs(m.gsd - 1.0) < 1e-6 and len(m) == 2
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _fake_paired_dataset(root, "rgb", "ndsm", n=3, h_val=40.0)
        f = aux.dfc2023_source(root, rgb_subdir="rgb", height_subdir="ndsm")
        assert f.name == "dfc2023" and abs(f.gsd - 0.5) < 1e-6 and len(f) == 3


def test_us3d_source_pairs_rgb_to_agl():
    # US3D naming: images/JAX_004_RGB.tif <-> truth/JAX_004_AGL.tif (one view per tile).
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rgb = (np.ones((3, 24, 24)) * 70).astype(np.uint8)
        h = (np.ones((24, 24)) * 9.0).astype(np.float32)
        _write_tif(root / "images" / "JAX_004_RGB.tif", rgb, 3)
        _write_tif(root / "truth" / "JAX_004_AGL.tif", h, 1)
        src = aux.us3d_source(root)                       # defaults: images/, truth/, RGB->AGL
        assert src.name == "us3d" and abs(src.gsd - 0.3) < 1e-6
        assert len(src) == 1, "US3D RGB->AGL pairing failed"


def test_build_aux_sources_wires_new_roots():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _fake_paired_dataset(root / "m4h", "rgb", "height", n=1, h_val=8.0)
        _fake_paired_dataset(root / "dfc", "rgb", "ndsm", n=1, h_val=20.0)
        srcs = aux.build_aux_sources(m4h_root=str(root / "m4h"), dfc_root=str(root / "dfc"))
        names = {s.name for s in srcs}
        assert names == {"m4heights", "dfc2023"}, names


def test_recursive_nested_pairing():
    # Open-Canopy nests by year, US3D by AOI — a flat glob would find 0 pairs.
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        rgb = (np.ones((3, 20, 20)) * 90).astype(np.uint8)
        h = (np.ones((20, 20)) * 6.0).astype(np.float32)
        _write_tif(root / "images" / "2021" / "regionA.tif", rgb, 3)
        _write_tif(root / "canopy_height" / "2021" / "regionA.tif", h, 1)
        src = aux.open_canopy_source(root, rgb_subdir="images", height_subdir="canopy_height")
        assert len(src) == 1, "nested (year subfolder) pair not found — glob not recursive"
        _, height = src.raw(0)
        assert abs(float(height.max()) - 6.0) < 1e-3


def test_split_aux_train_val_no_overlap():
    # G2: train/val split must not leak (disjoint tiles) and keep both sides.
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _fake_paired_dataset(root, "rgb", "ndsm", n=25, h_val=5.0)
        src = aux.geonrw_source(root, rgb_subdir="rgb", height_subdir="ndsm")
        tr, va = aux.split_aux_train_val([src], val_every=10)
        assert len(tr) == 1 and len(va) == 1
        tr_pairs, va_pairs = set(tr[0].pairs), set(va[0].pairs)
        assert tr_pairs.isdisjoint(va_pairs), "train/val overlap = leakage"
        assert len(tr_pairs) + len(va_pairs) == 25
        assert len(va[0]) == 3          # indices 0, 10, 20
        assert len(tr[0]) == 22


def test_split_aux_small_source_keeps_both_sides():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _fake_paired_dataset(root, "rgb", "ndsm", n=2, h_val=5.0)
        src = aux.geonrw_source(root, rgb_subdir="rgb", height_subdir="ndsm")
        tr, va = aux.split_aux_train_val([src], val_every=10)
        assert len(tr[0]) == 1 and len(va[0]) == 1   # never empties train


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
