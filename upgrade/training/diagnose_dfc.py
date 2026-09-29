"""Inspect a downloaded DFC2023 dataset before blending: confirm the optical+nDSM
folders and matched pairs, and CHECK THE nDSM UNITS — DFC heights should be ~0-200 m;
if the raster is in cm/dm the values are in the thousands and need a height_scale.

    python upgrade/training/diagnose_dfc.py --dfc-root D:\\Depth-Wizard\\DFC2023
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

UP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(UP / "training"))

import aux_datasets as aux  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dfc-root", required=True)
    a = ap.parse_args()

    src = aux.dfc2023_source(a.dfc_root)
    print(f"dfc2023: {len(src)} paired tiles (gsd={src.gsd})", flush=True)
    if len(src) == 0:
        rgb_s = [p.name for p in list(src.rgb_dir.rglob("*.tif"))[:4]] if src.rgb_dir.exists() else "MISSING"
        h_s = [p.name for p in list(src.height_dir.rglob("*.tif"))[:4]] if src.height_dir.exists() else "MISSING"
        print(f"  0 pairs — stems didn't match between folders.")
        print(f"  rgb_dir={src.rgb_dir}  sample={rgb_s}")
        print(f"  height_dir={src.height_dir}  sample={h_s}")
        print("  -> if names differ (e.g. x_rgb.tif vs x_ndsm.tif) pass rgb_token/height_token; "
              "if folder names are wrong, pass rgb_subdir/height_subdir.")
        return 1

    import rasterio
    rgb_p, h_p = src.pairs[0]
    with rasterio.open(rgb_p) as ds:
        print(f"  RGB  {Path(rgb_p).name}: {ds.width}x{ds.height} count={ds.count} "
              f"dtype={ds.dtypes[0]} tiled={ds.is_tiled}", flush=True)
    with rasterio.open(h_p) as ds:
        print(f"  nDSM {Path(h_p).name}: {ds.width}x{ds.height} dtype={ds.dtypes[0]} "
              f"nodata={ds.nodata} tiled={ds.is_tiled}", flush=True)

    _, h = src.raw(0)
    hv = h[np.isfinite(h)]
    print(f"  nDSM values after clean: min={hv.min():.1f} max={hv.max():.1f} "
          f"mean={hv.mean():.1f}  (expected ~0-200 m)", flush=True)
    if hv.max() > 500:
        print("  ** WARNING: max > 500 — heights look like cm/dm, not meters. Set a "
              "height_scale (0.01 for cm, 0.1 for dm) via dfc2023_source(..., height_scale=...) **")
    for i in range(min(3, len(src))):
        r, hh = src.raw(i)
        print(f"  [{i}] rgb{r.shape} h{hh.shape} hmax={float(hh.max()):.1f}", flush=True)

    print("\nOK — DFC2023 reads. Add --dfc-root to the blend (see the training command).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
