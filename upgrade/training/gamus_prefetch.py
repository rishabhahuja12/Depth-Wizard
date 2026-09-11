"""
One-time, resumable GAMUS prefetch — so training can then run fully OFFLINE.

Run this ONCE with internet (it downloads the whole train+val split, tens of GB,
to a local cache and writes a manifest per split). It is **resumable**: a tile
whose local RGB+AGL files already exist is skipped, so a dropped connection just
means "run it again to continue" — that existence check IS the dataset checkpoint.

    .venv/Scripts/python.exe upgrade/training/gamus_prefetch.py            # train + val
    .venv/Scripts/python.exe upgrade/training/gamus_prefetch.py --splits train

After it prints "COMPLETE" for a split, training/eval can run with --offline and
never touch the network again.

Pure helpers (agl_rel_for, local_paths, tiles_needing_download, build_manifest,
save/load_manifest) are unit-tested; the download itself is I/O.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

_REPO = "earthflow/GAMUS"
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "gamus_cache"


# --------------------------- pure helpers ---------------------------

def agl_rel_for(rgb_rel: str) -> str:
    return rgb_rel.replace("images/", "heights/").replace("_RGB.h5", "_AGL.h5")


def city_of(rgb_rel: str) -> str:
    return rgb_rel.split("/")[-1].split("_", 1)[0]


def tile_id_of(rgb_rel: str) -> str:
    return rgb_rel.split("/")[-1].replace("_RGB.h5", "")


def local_paths(rgb_rel: str, cache_dir: Path) -> tuple[Path, Path]:
    """Where hf_hub_download(local_dir=cache_dir) places the RGB and AGL files."""
    return cache_dir / rgb_rel, cache_dir / agl_rel_for(rgb_rel)


def tiles_needing_download(rgb_rels: list[str], cache_dir: Path) -> list[str]:
    """RGB rels whose local pair is NOT already fully present (the resume filter)."""
    out = []
    for rel in rgb_rels:
        rgb_p, agl_p = local_paths(rel, cache_dir)
        if not (rgb_p.exists() and agl_p.exists()):
            out.append(rel)
    return out


def build_manifest(rgb_rels: list[str], cache_dir: Path) -> dict:
    tiles = []
    for rel in rgb_rels:
        rgb_p, agl_p = local_paths(rel, cache_dir)
        tiles.append({
            "tile_id": tile_id_of(rel),
            "city": city_of(rel),
            "rgb_local": str(rgb_p),
            "agl_local": str(agl_p),
        })
    return {"repo": _REPO, "cache_dir": str(cache_dir), "count": len(tiles), "tiles": tiles}


def save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest_path(cache_dir: Path, split: str) -> Path:
    return cache_dir / f"manifest_{split}.json"


def _filelist_path(cache_dir: Path, split: str) -> Path:
    return cache_dir / f"filelist_{split}.json"


# --------------------------- I/O ---------------------------

def _list_tiles(split: str, cache_dir: Path) -> list[str]:
    """RGB rels for a split. Cached to disk so a resume needs no network to re-list."""
    cache_file = _filelist_path(cache_dir, split)
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    from huggingface_hub import list_repo_files
    all_files = list_repo_files(_REPO, repo_type="dataset")
    prefix = f"images/{split}/"
    rgb = sorted(f for f in all_files if f.startswith(prefix) and f.endswith("_RGB.h5"))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(rgb), encoding="utf-8")
    return rgb


def _download_pair(rgb_rel: str, cache_dir: Path, retries: int) -> bool:
    from huggingface_hub import hf_hub_download
    for attempt in range(retries):
        try:
            hf_hub_download(_REPO, rgb_rel, repo_type="dataset", local_dir=cache_dir)
            hf_hub_download(_REPO, agl_rel_for(rgb_rel), repo_type="dataset", local_dir=cache_dir)
            return True
        except Exception as e:  # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
            else:
                print(f"  FAILED {tile_id_of(rgb_rel)}: {type(e).__name__}: {e}")
                return False
    return False


def prefetch_split(split: str, cache_dir: Path, retries: int) -> bool:
    print(f"\n=== Prefetch split '{split}' -> {cache_dir}")
    rgb_rels = _list_tiles(split, cache_dir)
    needed = tiles_needing_download(rgb_rels, cache_dir)
    have = len(rgb_rels) - len(needed)
    print(f"  {len(rgb_rels)} tiles total | {have} already cached | {len(needed)} to fetch")

    ok = True
    for i, rel in enumerate(needed, 1):
        if not _download_pair(rel, cache_dir, retries):
            ok = False  # keep going; missing tiles just won't enter the manifest
        if i % 100 == 0 or i == len(needed):
            print(f"  fetched {i}/{len(needed)} ...", flush=True)

    # Manifest lists only tiles whose pair is actually present on disk.
    present = [r for r in rgb_rels if all(p.exists() for p in local_paths(r, cache_dir))]
    save_manifest(manifest_path(cache_dir, split), build_manifest(present, cache_dir))
    print(f"  {'COMPLETE' if ok and len(present) == len(rgb_rels) else 'PARTIAL'}: "
          f"{len(present)}/{len(rgb_rels)} tiles in manifest_{split}.json")
    return ok and len(present) == len(rgb_rels)


def warm_model(model_id: str) -> None:
    """Cache the model weights locally so offline training can load them (no network)."""
    from huggingface_hub import snapshot_download
    print(f"\n=== Caching model '{model_id}' (once) ...")
    for attempt in range(4):
        try:
            snapshot_download(repo_id=model_id)
            print("  model cached.")
            return
        except Exception as e:  # noqa: BLE001
            if attempt < 3:
                time.sleep(2.0 * (attempt + 1))
            else:
                print(f"  model cache FAILED: {type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="train,val", help="comma-separated splits to prefetch")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE), dest="cache_dir")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--model", default="depth-anything/Depth-Anything-V2-Large-hf",
                    help="model to cache for offline training")
    ap.add_argument("--no-model", action="store_true", dest="no_model",
                    help="skip caching the model (dataset only)")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    all_ok = True
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        all_ok = prefetch_split(split, cache, args.retries) and all_ok

    if not args.no_model:
        warm_model(args.model)

    print("\nAll set — you can now train with --offline (no internet needed)."
          if all_ok else
          "\nSome tiles missing — re-run this script (it resumes) until it says COMPLETE.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
