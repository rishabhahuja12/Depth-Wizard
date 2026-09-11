"""
Tests for P4 — off-nadir detect + FLAG (never correct/synthesize; research §3.3).

A lightweight, metadata-free cue: nadir aerial imagery has fairly isotropic edge
orientations; strong oblique/tilt introduces a dominant lean direction, raising an
anisotropy score. This is a coarse WARNING flag, not a measurement.

Run:
    .venv/Scripts/python.exe upgrade/inference/tests/test_off_nadir.py
"""
import sys
from pathlib import Path

import numpy as np

INF_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(INF_DIR))

import off_nadir as onx  # noqa: E402


def test_score_in_unit_range():
    rng = np.random.default_rng(0)
    img = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    s = onx.obliqueness_score(img)
    assert 0.0 <= s <= 1.0


def test_directional_edges_score_higher_than_isotropic():
    # Isotropic: random noise. Directional: strong parallel diagonal stripes
    # (a proxy for many buildings leaning the same way under off-nadir view).
    rng = np.random.default_rng(1)
    iso = (rng.random((128, 128, 3)) * 255).astype(np.uint8)

    yy, xx = np.mgrid[0:128, 0:128]
    stripes = (((xx + yy) % 10) < 3).astype(np.uint8) * 255
    directional = np.stack([stripes] * 3, axis=-1)

    assert onx.obliqueness_score(directional) > onx.obliqueness_score(iso)


def test_is_oblique_threshold_and_message():
    assert onx.is_oblique(0.9, threshold=0.5) is True
    assert onx.is_oblique(0.1, threshold=0.5) is False
    flag = onx.oblique_flag(0.9, threshold=0.5)
    assert flag["oblique"] is True and "oblique" in flag["message"].lower()


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
