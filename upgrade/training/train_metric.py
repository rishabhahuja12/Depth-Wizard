"""
P1 metric fine-tune loop — Depth Anything V2 Large regressing nDSM in METERS.

Runs on the CUDA workstation. Key differences from the old, broken
backend/training/train_gamus_full.py:
  * NO per-sample [0,1] normalization of pred/target — loss is in meters.
  * Backbone = Large (DINOv2-L + DPT), differential LR (enc 5e-6 / head 5e-5).
  * bf16 autocast + gradient checkpointing + grad-accumulation for 24 GB.
  * Early-stops on held-out MAE using the P0 metrics (evidence, not vibes).

Smoke-test the wiring on CPU (no GAMUS download, tiny random batch, small model):
    .venv/Scripts/python.exe upgrade/training/train_metric.py --smoke

Full run (workstation):
    .venv/Scripts/python.exe upgrade/training/train_metric.py
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch

TRAIN_DIR = Path(__file__).resolve().parent
UPGRADE_DIR = TRAIN_DIR.parent
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(UPGRADE_DIR / "evaluation"))

import metric_losses as ml  # noqa: E402

LARGE_MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"
SMALL_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
CKPT_DIR = UPGRADE_DIR / "checkpoints"


# --------------------------- pure helpers (unit-tested) ---------------------------

def lr_lambda(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    """Linear warmup for `warmup_epochs`, then cosine decay to 0."""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_param_groups(model, enc_lr: float, head_lr: float) -> list[dict]:
    """Differential LR: encoder ('backbone' in name) slow, DPT head fast."""
    enc = [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad]
    head = [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
    return [{"params": enc, "lr": enc_lr}, {"params": head, "lr": head_lr}]


# --------------------------- model + prediction ---------------------------

def load_model(model_id: str, device):
    from transformers import AutoModelForDepthEstimation
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    return model.to(device)


def predict_depth(model, images: torch.Tensor, out_hw) -> torch.Tensor:
    """Forward pass -> predicted depth interpolated to out_hw (H, W). Meters (once trained)."""
    pred = model(pixel_values=images).predicted_depth
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    pred = torch.nn.functional.interpolate(pred, size=out_hw, mode="bilinear", align_corners=False)
    return pred.squeeze(1)


# --------------------------- validation (held-out MAE) ---------------------------

@torch.no_grad()
def evaluate_mae(model, device, limit: int, crop: int) -> float:
    """Direct metric MAE on a held-out val subset — no calibration, no normalization."""
    import numpy as np
    import gamus_eval_loader as loader
    import metrics

    model.eval()
    errs = []
    for _tid, _city, rgb, agl in loader.iter_eval_tiles("val", limit=limit):
        x = _rgb_to_tensor(rgb, crop).to(device)
        pred = predict_depth(model, x.unsqueeze(0), agl.shape).squeeze(0).float().cpu().numpy()
        errs.append(metrics.mae(pred, agl))
    model.train()
    errs = [e for e in errs if e == e]  # drop nan
    return float(np.mean(errs)) if errs else float("nan")


def _rgb_to_tensor(rgb, crop: int):
    import metric_dataset as md
    from PIL import Image
    import numpy as np
    img = Image.fromarray(rgb).resize((crop, crop))
    return torch.from_numpy(md.normalize_rgb(np.array(img)))


# --------------------------- training ---------------------------

def train(args) -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  model: {args.model}")
    model = load_model(args.model, device)
    model.train()

    criterion = ml.MetricLoss(w_silog=args.w_silog, w_l1=args.w_l1).to(device)
    optim = torch.optim.AdamW(
        build_param_groups(model, args.enc_lr, args.head_lr), weight_decay=0.01
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda e: lr_lambda(e, args.warmup, args.epochs)
    )

    from torch.utils.data import DataLoader
    import metric_dataset as md
    dataset = md.MetricGAMUSDataset(split="train", crop=args.crop, augment=True)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)
    print(f"Train tiles: {len(dataset)} | batch {args.batch} x grad-accum {args.grad_accum}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_mae = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()
        running = 0.0
        optim.zero_grad(set_to_none=True)
        for i, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = targets.float().to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = predict_depth(model, images, targets.shape[-2:])
                loss = criterion(pred, targets) / args.grad_accum
            loss.backward()
            running += loss.item() * args.grad_accum
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)

        sched.step()
        val_mae = evaluate_mae(model, device, limit=args.val_tiles, crop=args.crop)
        dt = (time.time() - t0) / 60
        print(f"[epoch {epoch+1}/{args.epochs}] train_loss={running/max(len(loader),1):.4f} "
              f"val_MAE={val_mae:.3f} m  ({dt:.1f} min)")

        if val_mae == val_mae and val_mae < best_mae:
            best_mae = val_mae
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "val_mae_m": val_mae, "model_id": args.model, "metric": True},
                       CKPT_DIR / "metric_best.pth")
            print(f"  -> saved metric_best.pth (val MAE {val_mae:.3f} m)")

    print(f"Done. Best held-out MAE: {best_mae:.3f} m")
    return 0


def smoke(args) -> int:
    """Prove forward -> meters loss -> backward -> step wiring on CPU, no downloads
    beyond the (cached) small model, no GAMUS."""
    device = torch.device("cpu")
    print("SMOKE: loading small model on CPU...")
    model = load_model(SMALL_MODEL_ID, device)
    model.train()
    criterion = ml.MetricLoss()
    optim = torch.optim.AdamW(build_param_groups(model, 5e-6, 5e-5), weight_decay=0.01)

    s = args.smoke_size
    for step in range(2):
        images = torch.randn(1, 3, s, s)
        targets = torch.rand(1, s, s) * 40.0  # fake heights in meters
        pred = predict_depth(model, images, (s, s))
        loss = criterion(pred, targets)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        assert torch.isfinite(loss), "non-finite loss"
        print(f"  step {step+1}: loss={loss.item():.4f} (meters-scale), backward+step ok")
    print("SMOKE OK — pipeline wiring is sound.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=LARGE_MODEL_ID)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4, dest="grad_accum")
    ap.add_argument("--crop", type=int, default=518)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--enc-lr", type=float, default=5e-6, dest="enc_lr")
    ap.add_argument("--head-lr", type=float, default=5e-5, dest="head_lr")
    ap.add_argument("--w-silog", type=float, default=1.0, dest="w_silog")
    ap.add_argument("--w-l1", type=float, default=1.0, dest="w_l1")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-tiles", type=int, default=40, dest="val_tiles")
    ap.add_argument("--smoke", action="store_true", help="CPU wiring check, no GAMUS/GPU")
    ap.add_argument("--smoke-size", type=int, default=126, dest="smoke_size")
    args = ap.parse_args()
    return smoke(args) if args.smoke else train(args)


if __name__ == "__main__":
    sys.exit(main())
