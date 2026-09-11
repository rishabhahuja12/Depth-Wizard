"""
Known-answer tests for the pure helpers in train_metric.

The training loop itself is GPU glue (smoke-run with --smoke); only the
schedule math and the differential-LR param grouping are unit-tested here.

Run:
    .venv/Scripts/python.exe upgrade/training/tests/test_train_helpers.py
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

TRAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAIN_DIR))

import train_metric as tm  # noqa: E402


def _close(a, b, tol=1e-6, label=""):
    assert abs(a - b) <= tol, f"{label}: expected {b}, got {a}"


def test_lr_lambda_linear_warmup():
    _close(tm.lr_lambda(0, warmup_epochs=2, total_epochs=10), 0.5, label="warmup e0")
    _close(tm.lr_lambda(1, warmup_epochs=2, total_epochs=10), 1.0, label="warmup e1")


def test_lr_lambda_cosine_after_warmup():
    _close(tm.lr_lambda(2, warmup_epochs=2, total_epochs=10), 1.0, label="cosine start")
    # last epoch: progress = 7/8 -> 0.5*(1+cos(0.875*pi))
    expected = 0.5 * (1.0 + math.cos(0.875 * math.pi))
    _close(tm.lr_lambda(9, warmup_epochs=2, total_epochs=10), expected, tol=1e-6, label="cosine end")


def test_build_param_groups_splits_backbone_and_head():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(2, 2)  # weight + bias -> 2 params
            self.head = nn.Linear(2, 2)       # weight + bias -> 2 params

    model = Dummy()
    groups = tm.build_param_groups(model, enc_lr=5e-6, head_lr=5e-5)
    assert len(groups) == 2
    by_lr = {g["lr"]: g["params"] for g in groups}
    assert 5e-6 in by_lr and 5e-5 in by_lr, "missing an LR group"
    assert len(by_lr[5e-6]) == 2, "encoder group should hold the 2 backbone params"
    assert len(by_lr[5e-5]) == 2, "head group should hold the 2 head params"


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
