"""
DepthWizard MVP Training Script.
Fine-tunes Depth Anything V2 ViT-S on 200 GAMUS samples.
Expected time: ~2 minutes on RTX 4060.
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import time
import json
from pathlib import Path
from transformers import AutoModelForDepthEstimation

from training.losses import CombinedLoss
from training.dataset_gamus import GAMUSDataset
from app.config import (
    MODEL_ID, WEIGHTS_DIR, LOGS_DIR,
    TRAIN_SAMPLES, TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR
)


def train():
    assert torch.cuda.is_available(), "CUDA required for training!"
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 1. Dataset
    dataset = GAMUSDataset(split="train", max_samples=TRAIN_SAMPLES)
    loader = DataLoader(
        dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )

    # 2. Model
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    model.to(device)
    model.train()

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)
    scaler = GradScaler()
    criterion = CombinedLoss(alpha=0.5).to(device)

    # 4. Logging
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / "training_mvp.jsonl"

    best_loss = float("inf")
    total_start = time.time()

    print(f"\nStarting training: {len(loader)} batches/epoch, {TRAIN_EPOCHS} epochs\n")

    for epoch in range(1, TRAIN_EPOCHS + 1):
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(dtype=torch.float16):
                outputs = model(pixel_values=images)
                pred = outputs.predicted_depth

                # Resize prediction to match target
                if pred.shape[-2:] != targets.shape[-2:]:
                    pred = nn.functional.interpolate(
                        pred.unsqueeze(1), size=targets.shape[-2:],
                        mode="bilinear", align_corners=False
                    ).squeeze(1)

                # Normalize pred to similar range as target
                pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                target_norm = (targets - targets.min()) / (targets.max() - targets.min() + 1e-8)

                loss = criterion(pred_norm, target_norm)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # Log every batch
            log_entry = {
                "epoch": epoch, "batch": batch_idx,
                "loss": round(loss.item(), 5),
                "lr": optimizer.param_groups[0]["lr"],
            }
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{TRAIN_EPOCHS}] Loss: {avg_loss:.4f} Time: {elapsed:.1f}s")

        # Save checkpoint at epoch 5 and 10
        if epoch % 5 == 0:
            ckpt_path = WEIGHTS_DIR / f"checkpoint_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), WEIGHTS_DIR / "best_model.pth")
            print(f"  → Best model updated (loss: {best_loss:.4f})")

    # Save final
    torch.save(model.state_dict(), WEIGHTS_DIR / "final_model.pth")

    total_time = (time.time() - total_start) / 60
    print(f"\n{'='*50}")
    print(f"Training complete in {total_time:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Weights saved to: {WEIGHTS_DIR}")
    print(f"Logs saved to: {log_file}")


if __name__ == "__main__":
    train()
