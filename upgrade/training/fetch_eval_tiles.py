"""Fetch a sample of REAL held-out GAMUS val/test tiles into the local cache.

Why this exists: offline training with the leak fix refuses to fake val from train
tiles, so it needs genuine held-out `val` (and optionally `test`) tiles on disk.
The old gamus_prefetch is retired on some setups; this is the minimal replacement.

Run ONCE, ONLINE (do NOT set HF_HUB_OFFLINE), from the repo root:
    python upgrade/training/fetch_eval_tiles.py --splits val --n 60
    python upgrade/training/fetch_eval_tiles.py --splits val,test --n 60

Writes to upgrade/data/gamus_cache/ (train_metric's default --cache-dir), so a
later `--offline` run finds them via the local scan. Idempotent: already-cached
tiles are skipped, so a dropped connection just means "run it again".
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = "earthflow/GAMUS"
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "gamus_cache"


def _agl_for(rgb_rel: str) -> str:
    return rgb_rel.replace("images/", "heights/").replace("_RGB.h5", "_AGL.h5")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="val", help="comma-separated: val,test")
    ap.add_argument("--n", type=int, default=60, help="tiles per split (0 = all)")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE), dest="cache_dir")
    args = ap.parse_args()

    from huggingface_hub import list_repo_files, hf_hub_download

    cache = Path(args.cache_dir)
    all_files = list_repo_files(REPO, repo_type="dataset")

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        rgbs = sorted(f for f in all_files
                      if f.startswith(f"images/{split}/") and f.endswith("_RGB.h5"))
        if args.n > 0:
            rgbs = rgbs[:args.n]
        print(f"{split}: fetching {len(rgbs)} tiles -> {cache}", flush=True)
        for i, rgb in enumerate(rgbs, 1):
            for rel in (rgb, _agl_for(rgb)):
                if not (cache / rel).is_file():
                    hf_hub_download(REPO, rel, repo_type="dataset", local_dir=str(cache))
            if i % 20 == 0 or i == len(rgbs):
                print(f"  {i}/{len(rgbs)}", flush=True)

    print("done — offline validation now has real held-out tiles.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
