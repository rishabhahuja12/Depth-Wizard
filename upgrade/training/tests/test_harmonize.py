"""
Known-answer tests for multi-dataset harmonization (research §2.9).

Different sources (GAMUS 0.33 m, Open-Canopy 1.5 m, GBH ~3 m) must be resampled
to a common GSD and their labels forced to meters-above-ground before they can
share one height head. These pin the pure harmonization + mixing math.

Run:
    .venv/Scripts/python.exe upgrade/training/tests/test_harmonize.py
"""
import sys
from pathlib import Path

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAIN_DIR))

import harmonize as hz  # noqa: E402


def _close(a, b, tol=1e-6, label=""):
    assert abs(a - b) <= tol, f"{label}: expected {b}, got {a}"


def test_resample_factor_upsamples_coarser_source():
    # A 1.5 m source resampled to a 0.5 m grid must gain 3x pixels per axis.
    _close(hz.resample_factor(1.5, 0.5), 3.0, label="factor 1.5->0.5")
    _close(hz.resample_factor(0.33, 0.33), 1.0, label="factor same gsd")


def test_resampled_size_rounds():
    assert hz.resampled_size(100, 1.5, 0.5) == 300
    assert hz.resampled_size(512, 0.33, 0.33) == 512


def test_conform_meters_clamps_negatives_and_keeps_values():
    arr = np.array([[-2.0, 5.0], [10.0, np.nan]], dtype=np.float32)
    out = hz.conform_meters_above_ground(arr)
    assert out[0, 0] == 0.0            # negative -> ground
    assert out[1, 1] == 0.0            # nan -> 0
    assert out[0, 1] == 5.0 and out[1, 0] == 10.0


def test_sanitize_height_kills_positive_sentinel():
    # An undeclared positive sentinel (32767) must NOT survive as a 32km building.
    arr = np.array([[25.0, 32767.0], [np.inf, 3.0]], dtype=np.float32)
    out = hz.sanitize_height(arr)
    assert out[0, 0] == 25.0            # real height kept
    assert out[0, 1] == 0.0             # positive sentinel -> ground
    assert out[1, 0] == 0.0             # inf -> ground
    assert out[1, 1] == 3.0


def test_sanitize_height_nodata_and_scale():
    arr = np.array([[250.0, -9999.0], [100.0, 5.0]], dtype=np.float32)
    out = hz.sanitize_height(arr, nodata=-9999.0, scale=0.1)  # decimeters -> meters
    assert out[0, 1] == 0.0             # declared nodata -> ground (before scaling)
    assert abs(out[0, 0] - 25.0) < 1e-4  # 250 dm -> 25 m
    assert abs(out[1, 0] - 10.0) < 1e-4


def test_ndsm_from_dsm_dtm_subtracts_terrain():
    dsm = np.array([[110.0, 130.0], [105.0, 100.0]], dtype=np.float32)  # surface (AMSL)
    dtm = np.array([[100.0, 100.0], [108.0, 100.0]], dtype=np.float32)  # bare earth
    ndsm = hz.ndsm_from_dsm_dtm(dsm, dtm)
    assert ndsm[0, 0] == 10.0 and ndsm[0, 1] == 30.0  # building heights above ground
    assert ndsm[1, 0] == 0.0                          # DSM<DTM (noise) -> floored to 0
    assert ndsm[1, 1] == 0.0                          # flat ground


def test_balanced_source_plan_respects_weights():
    # Two sources, equal weight -> equal counts regardless of raw sizes (balance
    # so a big set doesn't drown a small one).
    plan = hz.balanced_source_plan(sizes={"gamus": 10000, "canopy": 200},
                                   weights={"gamus": 1.0, "canopy": 1.0}, total=100)
    assert plan["gamus"] == 50 and plan["canopy"] == 50


def test_balanced_source_plan_weight_ratio():
    plan = hz.balanced_source_plan(sizes={"a": 1000, "b": 1000},
                                   weights={"a": 3.0, "b": 1.0}, total=100)
    assert plan["a"] == 75 and plan["b"] == 25


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
