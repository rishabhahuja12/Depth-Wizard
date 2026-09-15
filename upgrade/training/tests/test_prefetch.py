"""
Tests for gamus_prefetch — now retired.

gamus_prefetch.py has been retired: training streams tiles directly from the
HuggingFace Hub mirror (earthflow/GAMUS) on-demand. No prefetch step is needed.

This test simply confirms the retired module can still be imported and that
running it prints a helpful retirement message without error.

Run:
    .venv/Scripts/python.exe upgrade/training/tests/test_prefetch.py
"""
import sys
import subprocess
from pathlib import Path

TRAIN_DIR = Path(__file__).resolve().parent.parent
PYTHON = sys.executable


def test_prefetch_module_imports_cleanly():
    """The retired module must import without raising."""
    sys.path.insert(0, str(TRAIN_DIR))
    import gamus_prefetch as pf  # noqa: F401
    assert callable(pf.main)


def test_prefetch_main_exits_zero():
    """Running the script directly must exit 0 and print a retirement notice."""
    script = TRAIN_DIR / "gamus_prefetch.py"
    result = subprocess.run([PYTHON, str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"non-zero exit: {result.stderr}"
    combined = result.stdout + result.stderr
    assert "no longer required" in combined.lower() or "retired" in combined.lower() or \
           "not needed" in combined.lower() or "prefetch is no longer" in combined.lower(), \
        f"expected retirement message, got:\n{combined}"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
        else:
            passed += 1
            print(f"ok    {t.__name__}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
