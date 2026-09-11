"""
Tests for P1b — absolute DSM via a real DEM base terrain.

DSM_absolute = DTM_base (real coarse DEM) + nDSM (AI metric height). Without this,
mountains render flat and the flood sim is meaningless (research §2A). These pin
the pure composition + DEM-to-grid upsampling.

Run:
    .venv/Scripts/python.exe upgrade/inference/tests/test_dem_base.py
"""
import sys
from pathlib import Path

import numpy as np

INF_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(INF_DIR))

import dem_base as dem  # noqa: E402


def test_compose_adds_base_and_ndsm():
    ndsm = np.array([[0.0, 10.0], [5.0, 0.0]], dtype=np.float32)
    base = np.array([[100.0, 100.0], [102.0, 102.0]], dtype=np.float32)
    out = dem.compose_absolute_dsm(ndsm, base)
    assert np.allclose(out, [[100.0, 110.0], [107.0, 102.0]])


def test_upsample_dem_matches_target_grid():
    coarse = np.array([[0.0, 10.0], [0.0, 10.0]], dtype=np.float32)  # 2x2
    up = dem.upsample_dem_to_grid(coarse, (4, 4))
    assert up.shape == (4, 4)
    # corners preserve the coarse values (a west-east ramp stays a ramp)
    assert up[0, 0] < up[0, -1]


def test_absolute_from_ndsm_upsamples_then_composes():
    ndsm = np.zeros((4, 4), dtype=np.float32)
    ndsm[1, 1] = 12.0  # a building
    coarse_dem = np.array([[50.0, 50.0], [60.0, 60.0]], dtype=np.float32)  # sloped ground
    out = dem.absolute_from_ndsm(ndsm, coarse_dem)
    assert out.shape == (4, 4)
    assert out[1, 1] > out[0, 0]              # building + its ground > bare ground
    assert out[-1, 0] > out[0, 0]             # southern ground is higher (slope preserved)


def test_flat_base_used_when_no_dem():
    ndsm = np.array([[0.0, 7.0]], dtype=np.float32)
    out = dem.absolute_from_ndsm(ndsm, None)  # non-georef -> flat base
    assert np.allclose(out, ndsm)             # equals nDSM (base = 0)


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
