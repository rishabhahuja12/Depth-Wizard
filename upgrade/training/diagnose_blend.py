"""Diagnose a stuck blend by testing the ACTUAL training read path (raw_window),
not a whole-tile read. Prints each raster's block layout — the fact that decides
whether a windowed read is fast (tiled/COG) or slow (striped: a window still
decompresses full-width strips).

    python upgrade/training/diagnose_blend.py --oc-root D:\\Depth-Wizard\\Open-Canopy

Read the output:
- is_tiled=True, small block_shapes (e.g. 256x256, 512x512): windowed reads are
  cheap; if raw_window is fast here but training still hangs, the problem is NOT
  the aux read.
- is_tiled=False (striped) and raw_window takes many seconds: THAT is the hang —
  the 40000x40000 mosaics must be converted to tiled COGs (or pre-tiled to small
  files) before training can window them cheaply.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

UP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(UP / "training"))

import aux_datasets            # noqa: E402
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
        print("  0 pairs — check folder layout (images/ + canopy_height/ *.tif).")
        return 1

    rgb_p, h_p = src.pairs[0]
    for label, p in (("RGB", rgb_p), ("HEIGHT", h_p)):
        with rasterio.open(p) as ds:
            prof = {k: ds.profile.get(k) for k in
                    ("width", "height", "count", "dtype", "compress", "tiled",
                     "blockxsize", "blockysize")}
            print(f"{label} {Path(p).name}: {prof}", flush=True)
            print(f"   is_tiled={ds.is_tiled}  block_shapes={ds.block_shapes[:1]}  "
                  f"overviews={ds.overviews(1)[:3]}", flush=True)

    win_px = int(round(a.crop * a.aux_gsd / max(src.gsd, 1e-6))) + 8
    print(f"\nwindow size the trainer reads: {win_px}x{win_px} native px", flush=True)
    for pos in [(0.5, 0.5), (0.1, 0.9), (0.9, 0.2)]:
        t = time.time()
        rgb, h = src.raw_window(0, win_px, pos=pos)
        print(f"  raw_window pos={pos}: rgb{rgb.shape} h{h.shape} in {time.time()-t:.2f}s", flush=True)

    print("\nfull MixedMetricDataset __getitem__(0) (window + resample + cell) ...", flush=True)
    t = time.time()
    ds = mdm.MixedMetricDataset([src], {src.name: 1.0}, target_gsd=a.aux_gsd,
                                crop=a.crop, total_per_epoch=len(src))
    x, y = ds[0]
    print(f"  item0: rgb{tuple(x.shape)} depth{tuple(y.shape)} in {time.time()-t:.2f}s", flush=True)

    print("\nVERDICT: if raw_window took <1s each, aux reads are fine — the training "
          "hang is elsewhere (tell me). If it took many seconds and is_tiled=False, "
          "the mosaics are striped and must be COG-converted / pre-tiled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
