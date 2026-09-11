"""
Known-answer tests for the pure logic in adapters (LoRA/DoRA).

The peft wrapping itself needs the library + a real model; only the pure
target-module selection and trainable-param accounting are unit-tested.

Run:
    .venv/Scripts/python.exe upgrade/training/tests/test_adapters.py
"""
import sys
from pathlib import Path

import torch.nn as nn

TRAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAIN_DIR))

import adapters  # noqa: E402


def test_lora_targets_are_attention_and_mlp_projections():
    targets = adapters.lora_target_modules()
    for name in ("query", "key", "value"):
        assert name in targets, f"missing attention proj {name}"
    assert any("fc" in t or "mlp" in t or "dense" in t for t in targets), "no MLP target"


def test_count_trainable_params():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(2, 2)   # 6 params (4 weight + 2 bias)
            self.b = nn.Linear(2, 2)   # 6 params

    m = Dummy()
    assert adapters.count_trainable(m) == 12
    for p in m.a.parameters():
        p.requires_grad = False
    assert adapters.count_trainable(m) == 6


def test_lora_config_dict_shape():
    cfg = adapters.lora_config_dict(r=16, alpha=32, dropout=0.05, use_dora=True)
    assert cfg["r"] == 16 and cfg["lora_alpha"] == 32
    assert cfg["lora_dropout"] == 0.05
    assert cfg["use_dora"] is True
    assert isinstance(cfg["target_modules"], list) and cfg["target_modules"]


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
