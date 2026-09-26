"""
P1 metric fine-tune loop — Depth Anything V2 Large regressing nDSM in METERS.

Runs on the CUDA workstation. Key differences from the old, broken
backend/training/train_gamus_full.py:
  * NO per-sample [0,1] normalization of pred/target — loss is in meters.
  * Backbone = Large (DINOv2-L + DPT), differential LR (enc 5e-6 / head 5e-5).
  * bf16 autocast + gradient checkpointing + grad-accumulation for 24 GB.
  * Early-stops on held-out MAE using the P0 metrics (evidence, not vibes).
  * Streams GAMUS tiles directly from HuggingFace Hub (no prefetch step needed);
    tiles are cached locally in upgrade/data/gamus_cache/ on first download.

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
sys.path.insert(0, str(UPGRADE_DIR))

import metric_losses as ml  # noqa: E402
from logging_config import get_logger, log_kv  # noqa: E402

LARGE_MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"
SMALL_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
CKPT_DIR = UPGRADE_DIR / "checkpoints"
DATA_CACHE = UPGRADE_DIR / "data" / "gamus_cache"


# --------------------------- pure helpers (unit-tested) ---------------------------

def lr_lambda(epoch: int, warmup_epochs: int, total_epochs: int) -> float:
    """Linear warmup for `warmup_epochs`, then cosine decay to 0."""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_param_groups(model, enc_lr: float, head_lr: float, lora_lr: float | None = None) -> list[dict]:
    """Differential LR: encoder ('backbone') slow, DPT head fast. LoRA adapters live
    INSIDE backbone modules (so 'backbone' is in their name) but must NOT inherit the
    tiny enc_lr — with B initialized to 0 they need ~1e-4 to move at all. Route any
    'lora_' param to its own (higher) lora_lr group."""
    named = list(model.named_parameters())
    lora = [p for n, p in named if "lora_" in n and p.requires_grad]
    enc = [p for n, p in named if "backbone" in n and "lora_" not in n and p.requires_grad]
    head = [p for n, p in named if "backbone" not in n and "lora_" not in n and p.requires_grad]
    groups = [{"params": enc, "lr": enc_lr}, {"params": head, "lr": head_lr}]
    if lora:
        groups.append({"params": lora, "lr": lora_lr if lora_lr is not None else head_lr})
    return groups


# --------------------------- model + prediction ---------------------------

def save_ckpt(path, model, optim, sched, epoch: int, best_mae: float) -> None:
    """Full training state so an interrupted weekend run can resume exactly."""
    import torch
    torch.save({
        "epoch": epoch,               # number of epochs COMPLETED
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optim.state_dict(),
        "scheduler_state_dict": sched.state_dict(),
        "best_mae": best_mae,
    }, path)


def load_ckpt(path, model, optim, sched):
    """Restore state saved by save_ckpt. Returns (start_epoch, best_mae)."""
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if optim is not None and ckpt.get("optimizer_state_dict"):
        optim.load_state_dict(ckpt["optimizer_state_dict"])
    if sched is not None and ckpt.get("scheduler_state_dict"):
        sched.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt.get("epoch", 0)), float(ckpt.get("best_mae", float("inf")))


def load_model(model_id: str, device, offline: bool = False):
    from transformers import AutoModelForDepthEstimation
    try:
        if offline:
            model = AutoModelForDepthEstimation.from_pretrained(model_id, local_files_only=True)
        else:
            model = AutoModelForDepthEstimation.from_pretrained(model_id)
    except Exception:
        model = AutoModelForDepthEstimation.from_pretrained(model_id, local_files_only=True)
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
def evaluate_mae(model, device, limit: int, crop: int, cache_dir,
                 gamus_root=None, offline: bool = False) -> float:
    """Direct metric MAE on a held-out val subset — no calibration, no normalization.
    Reads from local disk, cache, or Hub."""
    import numpy as np
    import metrics
    import gamus_eval_loader as loader

    src = loader.iter_eval_tiles("val", limit=limit, cache_dir=Path(cache_dir) if cache_dir else None,
                                 gamus_root=Path(gamus_root) if gamus_root else None, offline=offline)

    import metric_dataset as md
    model.eval()
    errs = []
    for _tid, _city, rgb, agl in src:
        # Evaluate on native-resolution crop cells (like training), NOT by squishing
        # the whole 1024 tile down to `crop` — that halved the GSD and skewed val MAE.
        cells = md.tile_grid(agl.shape[0], agl.shape[1], crop)
        if not cells:                        # tile smaller than crop: fall back to resize
            x = _rgb_to_tensor(rgb, crop).to(device)
            pred = predict_depth(model, x.unsqueeze(0), agl.shape).squeeze(0).float().cpu().numpy()
            errs.append(metrics.mae(pred, agl))
            continue
        for r0, c0 in cells:
            rgb_c, agl_c = rgb[r0:r0 + crop, c0:c0 + crop], agl[r0:r0 + crop, c0:c0 + crop]
            x = torch.from_numpy(md.normalize_rgb(rgb_c)).to(device)   # native res, no resize
            pred = predict_depth(model, x.unsqueeze(0), agl_c.shape).squeeze(0).float().cpu().numpy()
            errs.append(metrics.mae(pred, agl_c))
    model.train()
    errs = [e for e in errs if e == e]  # drop nan
    return float(np.mean(errs)) if errs else float("nan")


@torch.no_grad()
def evaluate_aux_mae(model, device, val_source, aux_gsd: float, crop: int, limit: int) -> float:
    """Metric MAE on a held-out aux source (G2), harmonized exactly like training
    (same resample-to-GSD + crop). Lets the gate measure the landscape this source
    adds (e.g. forests via Open-Canopy), which a GAMUS-urban val set can't see."""
    import numpy as np
    import metrics
    import multi_dataset as mdm

    total = min(len(val_source), limit) if limit else len(val_source)
    ds = mdm.MixedMetricDataset([val_source], {val_source.name: 1.0},
                                target_gsd=aux_gsd, crop=crop, total_per_epoch=total)
    model.eval()
    errs = []
    for k in range(len(ds)):
        rgb_t, depth_t = ds[k]
        pred = predict_depth(model, rgb_t.unsqueeze(0).to(device), depth_t.shape)
        errs.append(metrics.mae(pred.squeeze(0).float().cpu().numpy(), depth_t.numpy()))
    model.train()
    errs = [e for e in errs if e == e]
    return float(np.mean(errs)) if errs else float("nan")


def _rgb_to_tensor(rgb, crop: int):
    import metric_dataset as md
    from PIL import Image
    import numpy as np
    img = Image.fromarray(rgb).resize((crop, crop))
    return torch.from_numpy(md.normalize_rgb(np.array(img)))


# --------------------------- training ---------------------------

def parse_aux_weights(spec, sources) -> dict:
    """Per-source blend weights from a 'name=w,name2=w2' string; default 1.0 each.
    Lets a noisy source (e.g. `dfc2023=0.5`) get a smaller vote than a clean one,
    instead of every source counting equally (G1)."""
    weights = {s.name: 1.0 for s in sources}
    if spec:
        for part in str(spec).split(","):
            if "=" in part:
                name, val = part.split("=", 1)
                name = name.strip()
                if name in weights:
                    try:
                        weights[name] = max(0.0, float(val))
                    except ValueError:
                        print(f"  (ignored bad --aux-weights entry: {part!r})")
    return weights


def train(args) -> int:
    log = get_logger("train_metric")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_kv(log, "run start", device=str(device), model=args.model,
           epochs=args.epochs, warmup=args.warmup, batch=args.batch, grad_accum=args.grad_accum,
           crop=args.crop, enc_lr=args.enc_lr, head_lr=args.head_lr, lora=args.lora,
           blend=args.blend, aux_fraction=args.aux_fraction,
           w_silog=args.w_silog, w_l1=args.w_l1, w_grad=args.w_grad, w_lt=args.w_lt)

    offline = getattr(args, "offline", False)
    if offline:
        import os
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        print("Offline mode ENABLED: using local datasets and cached weights only.")

    gamus_root = getattr(args, "gamus_root", None)
    if gamus_root is None:
        default_root = UPGRADE_DIR.parent / "GAMUS"
        if default_root.exists():
            gamus_root = str(default_root)

    model = load_model(args.model, device, offline=offline)
    if args.lora:
        import adapters
        model = adapters.wrap_lora(model, r=args.lora_r, alpha=args.lora_alpha,
                                   dropout=args.lora_dropout, use_dora=args.dora)
        model.to(device)
    model.train()

    criterion = ml.MetricLoss(w_silog=args.w_silog, w_l1=args.w_l1,
                              w_grad=args.w_grad, w_lt=args.w_lt).to(device)
    stage = 1 + (args.w_grad > 0) + (args.w_lt > 0)
    print(f"Loss stage {stage}: silog={args.w_silog} l1={args.w_l1} "
          f"grad={args.w_grad} longtail={args.w_lt}")
    optim = torch.optim.AdamW(
        build_param_groups(model, args.enc_lr, args.head_lr,
                           lora_lr=getattr(args, "lora_lr", None) if args.lora else None),
        weight_decay=0.01,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda e: lr_lambda(e, args.warmup, args.epochs)
    )

    from torch.utils.data import DataLoader
    import metric_dataset as md
    dataset = md.MetricGAMUSDataset(split="train", crop=args.crop, augment=args.augment,
                                    cache_dir=Path(args.cache_dir),
                                    gamus_root=Path(gamus_root) if gamus_root else None,
                                    offline=offline,
                                    tile_size=args.tile_size,
                                    val_tiles=args.val_tiles)
    # Optional dataset blend: GAMUS (native tiling) + harmonized aux sources
    # (Open-Canopy / GBH) mixed in at a controlled fraction. Only after GAMUS-only
    # beats the baseline (research §2.8 sequencing).
    aux_val_sources = []                          # G2: held-out aux val (per-landscape)
    if args.blend:
        import aux_datasets
        import multi_dataset
        from torch.utils.data import ConcatDataset
        aux_sources = aux_datasets.build_aux_sources(
            oc_root=args.oc_root, gbh_root=args.gbh_root,
            geonrw_root=getattr(args, "geonrw_root", None),   # tolerate hand-built args
            m4h_root=getattr(args, "m4h_root", None),
            dfc_root=getattr(args, "dfc_root", None),
            us3d_root=getattr(args, "us3d_root", None))
        if aux_sources:
            # G2: hold out a slice of each aux source so the gate can measure the
            # landscape it adds (forest/diversity), not just GAMUS-urban val.
            aux_sources, aux_val_sources = aux_datasets.split_aux_train_val(aux_sources, val_every=10)
            gamus_len = len(dataset)
            aux_total = max(1, round(args.aux_fraction * gamus_len))
            weights = parse_aux_weights(getattr(args, "aux_weights", None), aux_sources)
            aux_mix = multi_dataset.MixedMetricDataset(aux_sources, weights, target_gsd=args.aux_gsd,
                                                       crop=args.crop, total_per_epoch=aux_total)
            dataset = ConcatDataset([dataset, aux_mix])
            print(f"Blend: GAMUS {gamus_len} + aux {len(aux_mix)} "
                  f"(train {[(s.name, len(s)) for s in aux_sources]}, "
                  f"val {[(s.name, len(s)) for s in aux_val_sources]}); weights={weights}")
        else:
            print("Blend requested but no aux sources found — training on GAMUS only.")

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)
    mode_str = "OFFLINE (Local)" if offline else "ONLINE (HF Hub)"
    print(f"Train samples: {len(dataset)} | batch {args.batch} x grad-accum {args.grad_accum} | {mode_str}")

    ckpt_dir = Path(getattr(args, "ckpt_dir", None) or CKPT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_path = ckpt_dir / "metric_last.pth"
    best_mae = float("inf")
    start_epoch = 0

    # Resume an interrupted run from the last full-state checkpoint.
    if args.resume and last_path.exists():
        start_epoch, best_mae = load_ckpt(last_path, model, optim, sched)
        print(f"Resumed from {last_path.name}: {start_epoch} epochs done, best MAE {best_mae:.3f}")

    for epoch in range(start_epoch, args.epochs):
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
        # G2: validate on GAMUS (urban) AND each held-out aux landscape, then gate
        # on the MEAN — so a blend that improves forests/diversity isn't rejected
        # just because GAMUS-urban MAE was flat.
        maes = {"gamus": evaluate_mae(model, device, limit=args.val_tiles, crop=args.crop,
                                      cache_dir=args.cache_dir, gamus_root=gamus_root, offline=offline)}
        for vs in aux_val_sources:
            maes[vs.name] = evaluate_aux_mae(model, device, vs, args.aux_gsd, args.crop,
                                             limit=args.val_tiles)
        finite = [v for v in maes.values() if v == v]
        val_mae = sum(finite) / len(finite) if finite else float("nan")  # landscape mean
        dt = (time.time() - t0) / 60
        per = " ".join(f"{k}={v:.2f}" for k, v in maes.items())
        log_kv(log, f"epoch {epoch+1}/{args.epochs}",
               train_loss=round(running / max(len(loader), 1), 4),
               val_mean_mae=round(val_mae, 3), per_landscape=f"[{per}]",
               lr=round(sched.get_last_lr()[0], 8), minutes=round(dt, 1))

        if val_mae == val_mae and val_mae < best_mae:
            best_mae = val_mae
            state = model.state_dict()
            if args.lora:
                # Serving loads a vanilla HF model — PEFT adapter keys won't match.
                # Merge the adapters into the base weights for the saved checkpoint,
                # then unmerge so training continues on the adapters.
                try:
                    model.merge_adapter()
                    state = model.get_base_model().state_dict()
                    model.unmerge_adapter()
                except Exception as e:  # noqa: BLE001
                    print(f"  LoRA merge for serving failed ({e}); saving adapter state as-is")
            torch.save({"epoch": epoch + 1, "model_state_dict": state,
                        "val_mae_m": val_mae, "val_maes_by_landscape": maes,
                        "model_id": args.model, "metric": True},
                       ckpt_dir / "metric_best.pth")
            log_kv(log, "saved metric_best.pth", val_mae=round(val_mae, 3), epoch=epoch + 1)

        # Always save full state so the run can resume after any interruption.
        save_ckpt(last_path, model, optim, sched, epoch + 1, best_mae)

    log_kv(log, "run done", best_val_mae=round(best_mae, 3), epochs=args.epochs)
    return 0


def smoke(args) -> int:
    """Prove forward -> meters loss -> backward -> step wiring on CPU, no downloads
    beyond the cached model, no GAMUS."""
    device = torch.device("cpu")
    print("SMOKE: loading model on CPU...")
    try:
        model = load_model(SMALL_MODEL_ID, device, offline=args.offline)
    except Exception:
        print(f"SMOKE: {SMALL_MODEL_ID} not cached locally; falling back to {LARGE_MODEL_ID}...")
        model = load_model(LARGE_MODEL_ID, device, offline=args.offline)
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
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8, dest="grad_accum")  # eff. batch 16
    ap.add_argument("--crop", type=int, default=504,
                    help="deterministic tile-cell size — a multiple of 14 (DINOv2 patch); "
                         "504=36x14 keeps 4 cells per 1024 tile with no patch-boundary interp")
    ap.add_argument("--augment", action="store_true",
                    help="opt-in flip/rotate aug (default OFF — zero-augmentation, raw tiles)")
    # Adapters (fallback / Giant-enabler; full FT is the primary path).
    ap.add_argument("--lora", action="store_true", help="fine-tune via LoRA adapters instead of full FT")
    ap.add_argument("--dora", action="store_true", help="use DoRA variant (implies LoRA)")
    ap.add_argument("--lora-r", type=int, default=16, dest="lora_r")
    ap.add_argument("--lora-alpha", type=int, default=32, dest="lora_alpha")
    ap.add_argument("--lora-dropout", type=float, default=0.05, dest="lora_dropout")
    ap.add_argument("--lora-lr", type=float, default=2e-4, dest="lora_lr",
                    help="LR for LoRA adapters (they need ~1e-4, not the tiny enc_lr)")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--enc-lr", type=float, default=5e-6, dest="enc_lr")
    ap.add_argument("--head-lr", type=float, default=5e-5, dest="head_lr")
    ap.add_argument("--w-silog", type=float, default=1.0, dest="w_silog")
    ap.add_argument("--w-l1", type=float, default=1.0, dest="w_l1")
    ap.add_argument("--w-grad", type=float, default=0.0, dest="w_grad",
                    help="Stage 2: Sobel edge loss weight (0 = off)")
    ap.add_argument("--w-lt", type=float, default=0.0, dest="w_lt",
                    help="Stage 3: long-tail tall-structure weight (0 = off)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--val-tiles", type=int, default=40, dest="val_tiles")
    ap.add_argument("--resume", action="store_true",
                    help="resume from upgrade/checkpoints/metric_last.pth if present")
    ap.add_argument("--tile-size", type=int, default=1024, dest="tile_size",
                    help="source tile size for deterministic cell grid (GAMUS = 1024)")
    # Dataset blend (Open-Canopy / aux sources) — only after GAMUS-only wins.
    ap.add_argument("--blend", action="store_true", help="mix aux sources into GAMUS")
    ap.add_argument("--oc-root", default=None, dest="oc_root", help="Open-Canopy download root")
    ap.add_argument("--gbh-root", default=None, dest="gbh_root", help="GBH download root")
    ap.add_argument("--geonrw-root", default=None, dest="geonrw_root",
                    help="GeoNRW root (ON HOLD; height_subdir must be DERIVED nDSM — see derive_ndsm)")
    ap.add_argument("--m4h-root", default=None, dest="m4h_root",
                    help="M4Heights root (extracted 1 m aerial rgb/ + height/)")
    ap.add_argument("--dfc-root", default=None, dest="dfc_root",
                    help="DFC2023 Track 2 root (train split; rgb/ + ndsm/)")
    ap.add_argument("--us3d-root", default=None, dest="us3d_root",
                    help="US3D/DFC2019 root (FALLBACK; images/ *_RGB.tif + truth/ *_AGL.tif)")
    ap.add_argument("--aux-fraction", type=float, default=0.3, dest="aux_fraction",
                    help="aux samples as a fraction of GAMUS length")
    ap.add_argument("--aux-weights", default=None, dest="aux_weights",
                    help="per-source blend weights, e.g. 'open_canopy=1.0,dfc2023=0.5'")
    ap.add_argument("--aux-gsd", type=float, default=0.5, dest="aux_gsd",
                    help="common GSD (m) to harmonize aux sources to")
    ap.add_argument("--cache-dir", default=str(DATA_CACHE), dest="cache_dir",
                    help="local tile cache dir; hf_hub_download writes here, reused on repeated runs")
    ap.add_argument("--gamus-root", default=None, dest="gamus_root",
                    help="local GAMUS dataset root directory (e.g. D:\\Depth-Wizard\\GAMUS; defaults to ./GAMUS if present)")
    ap.add_argument("--offline", action="store_true",
                    help="run 100%% offline using local datasets and cached weights only")
    ap.add_argument("--ckpt-dir", default=str(CKPT_DIR), dest="ckpt_dir",
                    help="where to write metric_best.pth / metric_last.pth")
    ap.add_argument("--smoke", action="store_true", help="CPU wiring check, no GAMUS/GPU")
    ap.add_argument("--smoke-size", type=int, default=126, dest="smoke_size")
    args = ap.parse_args()
    if args.dora:
        args.lora = True  # DoRA is a LoRA variant

    return smoke(args) if args.smoke else train(args)


if __name__ == "__main__":
    sys.exit(main())
