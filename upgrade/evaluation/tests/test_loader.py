"""
Known-answer tests for the pure parsing logic in gamus_eval_loader.

Run:
    .venv/Scripts/python.exe upgrade/evaluation/tests/test_loader.py

Only the pure functions are tested here (city parsing / grouping). The
HuggingFace download + h5 reading is I/O glue exercised by the smoke run,
not unit-tested.
"""
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_DIR))

import gamus_eval_loader as loader  # noqa: E402


def test_city_of_extracts_prefix_before_first_underscore():
    assert loader.city_of("images/test/DC_03_26_RGB.h5") == "DC"
    assert loader.city_of("images/val/PHL_6922_RGB.h5") == "PHL"
    assert loader.city_of("JAX_10_02_RGB.h5") == "JAX"


def test_agl_path_mirrors_rgb_path():
    rgb = "images/test/DC_03_26_RGB.h5"
    assert loader.agl_path_for(rgb) == "heights/test/DC_03_26_AGL.h5"


def test_group_by_city_counts_tiles():
    rgb_files = [
        "images/test/DC_01_RGB.h5",
        "images/test/DC_02_RGB.h5",
        "images/test/JAX_01_RGB.h5",
    ]
    groups = loader.group_by_city(rgb_files)
    assert groups["DC"] == 2
    assert groups["JAX"] == 1


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
