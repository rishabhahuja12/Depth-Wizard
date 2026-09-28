"""Diagnose a stuck blend: read Open-Canopy tiles directly (NO DataLoader) and time
each step, so a hang localizes to rasterio read vs. whole-tile resample vs. cell crop.

    python upgrade/training/diagnose_blend.py --oc-root D:\\Depth-Wizard\\Open-Canopy

If every step prints quickly -> the hang is the DataLoader (use --workers 0).
If it stalls at 'resample' or a specific tile prints a huge WxH -> the tile is too
big to resample whole; that's a data/code issue, not the loader.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

UP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(UP / "training"))

import aux_datasets            # noqa: E402
import harmonize as hz         # noqa: E402
import multi_dataset as mdm    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oc-root", required=True)
    ap.add_argument("--aux-gsd", type=float, default=0.5)
    ap.add_argument("--crop", type=int, default=504)
    a = ap.parse_args()

    import rasterio

    src = aux_datasets.open_canopy_source(a.oc_root)
    print(f"open_canopy: {len(src)} paired tiles (gsd={src.gsd})", flush=True)
    if len(src) == 0:
        print("  0 pairs — check the folder layout (images/ + canopy_height/ *.tif).")
        return 1

    print("tile dimensions (first 5):", flush=True)
    for i, (rgb_p, h_p) in enumerate(src.pairs[:5]):
        with rasterio.open(rgb_p) as ds:
            rw, rh, rc = ds.width, ds.height, ds.count
        with rasterio.open(h_p) as ds:
            hw, hh = ds.width, ds.height
        print(f"  [{i}] rgb {rw}x{rh}x{rc} ({os.path.getsize(rgb_p)//1_000_000}MB)  "
              f"height {hw}x{hh} ({os.path.getsize(h_p)//1_000_000}MB)", flush=True)

    print("reading raw(0) ...", flush=True)
    t = time.time()
    rgb, h = src.raw(0)
    print(f"  raw(0): rgb{rgb.shape} h{h.shape} in {time.time()-t:.1f}s", flush=True)

    factor = src.gsd / a.aux_gsd
    print(f"resample whole tile {src.gsd}m -> {a.aux_gsd}m (x{factor:.1f} per axis, "
          f"~x{factor**2:.0f} pixels) ...", flush=True)
    t = time.time()
    rgb2 = hz.resample_to_gsd(rgb, src.gsd, a.aux_gsd, order=1)
    print(f"  resampled rgb {rgb.shape}->{rgb2.shape} in {time.time()-t:.1f}s", flush=True)

    print("full MixedMetricDataset __getitem__(0) ...", flush=True)
    t = time.time()
    ds = mdm.MixedMetricDataset([src], {src.name: 1.0}, target_gsd=a.aux_gsd,
                                crop=a.crop, total_per_epoch=len(src))
    x, y = ds[0]
    print(f"  item0: rgb{tuple(x.shape)} depth{tuple(y.shape)} in {time.time()-t:.1f}s", flush=True)
    print("\nOK — aux reading works. If all fast, the training hang is the DataLoader "
          "(use --workers 0). If a step was very slow, that tile is too big to resample whole.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
