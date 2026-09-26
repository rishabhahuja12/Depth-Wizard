"""Honest held-out eval — score a metric checkpoint on the GAMUS **test** split the
SAME way training computes val MAE (raw predicted_depth vs AGL meters, native-res
crop cells). So the aggregate number is directly comparable to the training log's
`val_mean_mae` (~0.64) — but on tiles the model never trained on. Also reports the
building-only (tall) MAE and the scale-invariant aligned MAE, so you can see how
much of the error is scale vs. shape.

    .venv/Scripts/python.exe upgrade/evaluation/eval_test.py --limit 50
    .venv/Scripts/python.exe upgrade/evaluation/eval_test.py --split test --limit 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

UP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(UP / "training"))
sys.path.insert(0, str(UP / "evaluation"))

import metrics  # noqa: E402
import gamus_eval_loader as loader  # noqa: E402
import metric_dataset as md  # noqa: E402
from train_metric import load_model, predict_depth, LARGE_MODEL_ID  # noqa: E402


def _mean(xs):
    xs = [v for v in xs if v == v]  # drop NaN
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(UP / "checkpoints" / "metric_best.pth"))
    ap.add_argument("--split", default="test", help="held-out split (test = never trained on)")
    ap.add_argument("--limit", type=int, default=50, help="cap tiles (start small)")
    ap.add_argument("--crop", type=int, default=504)
    ap.add_argument("--model", default=LARGE_MODEL_ID)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    try:  # safe by default; our checkpoints are plain tensors + primitives
        ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    except Exception:
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)  # legacy fallback
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    res = model.load_state_dict(sd, strict=False)
    total = len(model.state_dict())
    matched = total - len(getattr(res, "missing_keys", []))
    ck_mae = ck.get("val_mae_m") if isinstance(ck, dict) else None
    print(f"loaded {args.ckpt}: matched {matched}/{total} keys | ckpt val_mae_m={ck_mae}")
    if matched < 0.5 * total:
        print("  WARNING: <50% keys matched — wrong architecture/checkpoint? Numbers unreliable.")
    model.eval()

    agg, tall, aligned = [], [], []
    n = 0
    for tid, city, rgb, agl in loader.iter_eval_tiles(args.split, args.limit):
        cells = md.tile_grid(agl.shape[0], agl.shape[1], args.crop) or [(0, 0)]
        for r0, c0 in cells:
            rgb_c = rgb[r0:r0 + args.crop, c0:c0 + args.crop]
            agl_c = agl[r0:r0 + args.crop, c0:c0 + args.crop]
            x = torch.from_numpy(md.normalize_rgb(rgb_c)).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = predict_depth(model, x, agl_c.shape).squeeze(0).float().cpu().numpy()
            agg.append(metrics.mae(pred, agl_c))
            tall.append(metrics.tall_structure_mae(pred, agl_c))          # buildings only (>15 m)
            s, t = metrics.affine_fit(pred, agl_c)                        # best-fit scale+shift
            aligned.append(metrics.mae(s * pred + t, agl_c))
        n += 1
        print(f"  [{n}] {tid} ({city})")

    print(f"\n==== {args.split.upper()} split (held-out) — {n} tiles, crop {args.crop} ====")
    print(f"aggregate MAE   = {_mean(agg):.3f} m   <- compare directly to training val_mean_mae (~0.64)")
    print(f"tall MAE (>15m) = {_mean(tall):.3f} m   <- buildings only, the number that matters")
    print(f"aligned MAE     = {_mean(aligned):.3f} m   <- after best-fit scale/shift (shape-only error)")
    print("\nIf aggregate here is much higher than 0.64, the val number was in-distribution optimism.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
