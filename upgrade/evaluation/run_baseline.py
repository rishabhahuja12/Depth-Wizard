"""
P0 baseline runner — measures the CURRENT DepthWizard pipeline on the held-out
GAMUS test split and writes the before-table.

    .venv/Scripts/python.exe upgrade/evaluation/run_baseline.py --limit 3     # smoke test
    .venv/Scripts/python.exe upgrade/evaluation/run_baseline.py               # full split

Every tile is scored two ways (see report.py):
  * scale-invariant  — best-fit aligned; the model's shape quality
  * metric as-shipped — through the real calibration; exposes the meters gap

Depends on backend/ (one-directional; backend never imports this).
"""
from __future__ import annotations

import argparse
import sys
import statistics
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(REPO_ROOT / "backend"))

import metrics  # noqa: E402
import gamus_eval_loader as loader  # noqa: E402
import report  # noqa: E402

EDGE_THRESH = 0.05  # on range-normalized [0,1] maps


def _norm01(a: np.ndarray) -> np.ndarray:
    m = metrics.valid_mask(a, a)
    if not np.any(m):
        return np.zeros_like(a, dtype=np.float64)
    lo, hi = float(a[m].min()), float(a[m].max())
    if hi - lo < 1e-9:
        return np.zeros_like(a, dtype=np.float64)
    return np.clip((a.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)


def _scale_invariant_block(rel: np.ndarray, gt: np.ndarray) -> dict:
    s, t = metrics.affine_fit(rel, gt)
    aligned = s * rel + t
    d1, _, _ = metrics.delta_thresholds(aligned, gt)
    return {
        "aligned_rmse": metrics.affine_aligned_rmse(rel, gt),
        "si_rmse": metrics.si_rmse(rel, gt),
        "aligned_mae": metrics.mae(aligned, gt),
        "aligned_delta1": d1,
        "aligned_tall_mae": metrics.tall_structure_mae(aligned, gt),
        "boundary_f": metrics.boundary_f_score(_norm01(rel), _norm01(gt), EDGE_THRESH),
    }


def _metric_as_app_block(rel: np.ndarray, gt: np.ndarray, gsd: float) -> dict:
    from app.services.calibration import calibrate_depth

    cal = calibrate_depth(rel, is_georef=True, gsd=gsd)
    dsm = cal.dsm
    d1, _, _ = metrics.delta_thresholds(dsm, gt)
    return {
        "rmse": metrics.rmse(dsm, gt),
        "mae": metrics.mae(dsm, gt),
        "delta1": d1,
        "tall_mae": metrics.tall_structure_mae(dsm, gt),
        "boundary_f": metrics.boundary_f_score(_norm01(dsm), _norm01(gt), EDGE_THRESH),
    }


def _mean_block(blocks: list[dict]) -> dict:
    if not blocks:
        return {}
    keys = blocks[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in blocks if b.get(k) is not None and not np.isnan(b[k])]
        out[k] = float(statistics.fmean(vals)) if vals else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="cap tiles (smoke test); default = full split")
    ap.add_argument("--gsd", type=float, default=0.33,
                    help="GAMUS meters/pixel (VERIFIED 0.33 m — arXiv:2305.14914)")
    ap.add_argument("--out", default=str(REPO_ROOT / "upgrade" / "outputs"))
    args = ap.parse_args()

    from app.services.depth_estimator import DepthEstimator

    print(f"Loading current pipeline model...")
    estimator = DepthEstimator()

    si_all, app_all = [], []
    si_by_city: dict[str, list] = {}
    app_by_city: dict[str, list] = {}
    n_tiles = 0
    tiles_by_city: dict[str, int] = {}

    for tile_id, city, rgb, agl in loader.iter_eval_tiles(args.split, args.limit):
        rel = estimator.predict(Image.fromarray(rgb))
        if rel.shape != agl.shape:
            print(f"  skip {tile_id}: shape {rel.shape} vs {agl.shape}")
            continue

        si = _scale_invariant_block(rel, agl)
        app = _metric_as_app_block(rel, agl, args.gsd)

        si_all.append(si); app_all.append(app)
        si_by_city.setdefault(city, []).append(si)
        app_by_city.setdefault(city, []).append(app)
        tiles_by_city[city] = tiles_by_city.get(city, 0) + 1
        n_tiles += 1
        print(f"  [{n_tiles}] {tile_id} ({city})  aligned_rmse={si['aligned_rmse']:.2f}m  "
              f"as-shipped_rmse={app['rmse']:.2f}m")

    if n_tiles == 0:
        print("No tiles evaluated — aborting.")
        return 1

    results = {
        "config": {
            "split": args.split,
            "tiles_evaluated": n_tiles,
            "limit": args.limit if args.limit is not None else "full split",
            "gsd_m": f"{args.gsd} (verified — arXiv:2305.14914)",
            "model_id": __import__("app.config", fromlist=["MODEL_ID"]).MODEL_ID,
            "device": str(estimator.device),
        },
        "overall": {
            "n_tiles": n_tiles,
            "scale_invariant": _mean_block(si_all),
            "metric_as_app": _mean_block(app_all),
        },
        "per_city": {
            city: {
                "n_tiles": tiles_by_city[city],
                "scale_invariant": _mean_block(si_by_city[city]),
                "metric_as_app": _mean_block(app_by_city[city]),
            }
            for city in tiles_by_city
        },
    }

    md_path, csv_path = report.write_reports(results, Path(args.out))
    print(f"\nWrote:\n  {md_path}\n  {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
