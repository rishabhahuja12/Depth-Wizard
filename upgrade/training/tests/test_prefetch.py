"""
Known-answer tests for the pure logic in gamus_prefetch.

The resumable-download "checkpoint" is: a tile whose local RGB+AGL files already
exist is skipped. These tests pin that skip logic and the manifest round-trip so a
half-finished prefetch can safely resume and an offline run can read the result.

Run:
    .venv/Scripts/python.exe upgrade/training/tests/test_prefetch.py
"""
import sys
import tempfile
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAIN_DIR))

import gamus_prefetch as pf  # noqa: E402


def test_agl_rel_mirrors_rgb_rel():
    assert pf.agl_rel_for("images/train/DC_03_26_RGB.h5") == "heights/train/DC_03_26_AGL.h5"


def test_local_paths_place_under_cache_dir():
    rgb, agl = pf.local_paths("images/train/DC_1_RGB.h5", Path("/cache"))
    assert rgb == Path("/cache/images/train/DC_1_RGB.h5")
    assert agl == Path("/cache/heights/train/DC_1_AGL.h5")


def test_tiles_needing_download_skips_completed_pairs():
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d)
        rgbs = ["images/train/A_RGB.h5", "images/train/B_RGB.h5"]
        # Mark A complete by creating BOTH its local files.
        for rel in (rgbs[0], pf.agl_rel_for(rgbs[0])):
            p = cache / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
        needed = pf.tiles_needing_download(rgbs, cache)
        assert needed == ["images/train/B_RGB.h5"], f"got {needed}"


def test_tiles_needing_download_requires_both_files():
    with tempfile.TemporaryDirectory() as d:
        cache = Path(d)
        rel = "images/train/A_RGB.h5"
        # Only the RGB present, AGL missing -> still needs download.
        p = cache / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        assert pf.tiles_needing_download([rel], cache) == [rel]


def test_build_manifest_entries():
    m = pf.build_manifest(["images/test/JAX_9_RGB.h5"], Path("/c"))
    assert m["count"] == 1
    e = m["tiles"][0]
    assert e["tile_id"] == "JAX_9" and e["city"] == "JAX"
    assert e["rgb_local"] == str(Path("/c/images/test/JAX_9_RGB.h5"))
    assert e["agl_local"] == str(Path("/c/heights/test/JAX_9_AGL.h5"))


def test_manifest_round_trip():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "manifest_train.json"
        manifest = pf.build_manifest(["images/train/DC_1_RGB.h5"], Path("/c"))
        pf.save_manifest(path, manifest)
        assert pf.load_manifest(path) == manifest


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
