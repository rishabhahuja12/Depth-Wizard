"""
gamus_prefetch.py — RETIRED

Training and evaluation now stream GAMUS tiles directly from the HuggingFace Hub
mirror (earthflow/GAMUS) via hf_hub_download, with automatic per-tile local caching.
No prefetch step is needed before running train_metric.py.

How caching works:
  * The first time a tile is needed, hf_hub_download fetches it from HF Hub and
    writes it to upgrade/data/gamus_cache/ (or your --cache-dir).
  * Every subsequent run reuses the cached file — no re-download.
  * If a tile is missing mid-run (e.g. a new training pass before all tiles are
    cached), it is downloaded on-demand with automatic retries.

To start training immediately (tiles are fetched as needed):
    .venv/Scripts/python.exe upgrade/training/train_metric.py --resume ^
      --epochs 50 --warmup 3 ^
      --enc-lr 2.5e-6 --head-lr 2.5e-5 ^
      --batch 2 --grad-accum 8 --crop 504 ^
      --max-vram-frac 0.5 --throttle-sleep 0.15 ^
      --workers 4 >> upgrade/outputs/train_stage1.log

Smoke-test (CPU, no GPU, no data download):
    .venv/Scripts/python.exe upgrade/training/train_metric.py --smoke
"""

import sys


def main() -> int:
    print(__doc__)
    print("Nothing to do — prefetch is no longer required.")
    print("Run train_metric.py directly to start training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
