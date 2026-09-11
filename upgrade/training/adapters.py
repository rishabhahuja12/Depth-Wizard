"""
LoRA / DoRA adapters for the metric fine-tune.

Role (research §2.4): the *fallback* / A-B lever and the enabler for testing
Giant on 24 GB — NOT the primary path (full fine-tune on Large is). LoRA freezes
the backbone and learns small low-rank deltas, so it fits easily, trains fast, and
de-risks catastrophic forgetting, at a slight ceiling cost vs full FT.

Uses `peft` (a code dependency, no network at runtime — fine for offline runs).
Pure helpers (targets, config, counting) are unit-tested; the wrapping needs peft
+ a real model and is exercised on the workstation.
"""
from __future__ import annotations


def lora_target_modules() -> list[str]:
    """Module-name substrings peft should adapt: the DINOv2 attention projections
    and the MLP/DPT linear layers. peft matches these as suffixes/substrings."""
    return ["query", "key", "value", "dense", "fc1", "fc2", "projection"]


def lora_config_dict(r: int = 16, alpha: int = 32, dropout: float = 0.05,
                     use_dora: bool = False) -> dict:
    """Plain dict of the LoRA/DoRA config (kept library-agnostic for testing)."""
    return {
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "use_dora": use_dora,
        "target_modules": lora_target_modules(),
        "bias": "none",
    }


def count_trainable(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total(model) -> int:
    return sum(p.numel() for p in model.parameters())


def wrap_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05,
              use_dora: bool = False):
    """Wrap a HF depth model with LoRA/DoRA. Returns the peft model.

    Raises a clear error (not an obscure ImportError) if peft is absent so the
    workstation setup is unambiguous.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as e:  # noqa: BLE001
        raise RuntimeError(
            "LoRA requested but `peft` is not installed. Run: pip install peft"
        ) from e

    cfg = lora_config_dict(r, alpha, dropout, use_dora)
    lora = LoraConfig(
        r=cfg["r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], bias=cfg["bias"], use_dora=cfg["use_dora"],
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    peft_model = get_peft_model(model, lora)
    trainable, total = count_trainable(peft_model), count_total(peft_model)
    print(f"LoRA{'/DoRA' if use_dora else ''} r={r} α={alpha}: "
          f"{trainable/1e6:.2f}M trainable / {total/1e6:.1f}M total "
          f"({100*trainable/total:.2f}%)")
    return peft_model
