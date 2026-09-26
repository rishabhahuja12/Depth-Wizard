"""
Depth-Wizard — Open-Canopy Resilient Downloader
Calls the bulletproof paired downloader in download_all_aux_datasets.py
"""
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_all_aux_datasets import download_open_canopy

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Open-Canopy 2023 paired dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of paired tiles (e.g. --limit 5). Default: all 18.")
    parser.add_argument("--token", type=str, default=None, help="Optional Hugging Face access token.")
    args = parser.parse_args()

    success = download_open_canopy(limit=args.limit, token=args.token)
    sys.exit(0 if success else 1)
