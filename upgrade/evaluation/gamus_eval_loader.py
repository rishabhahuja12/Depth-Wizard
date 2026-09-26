"""
Held-out GAMUS loader for the P0 baseline.

The current model was fine-tuned on the `train` split, so the `test` split is
genuinely unseen. GAMUS filenames encode the city as the token before the first
underscore (e.g. DC_03_26_RGB.h5 -> "DC"), which lets us report a per-city
breakdown for an honest "unseen geography" signal.

RGB tiles are `images/<split>/*_RGB.h5`; matching heights (nDSM in meters,
"above ground level") are `heights/<split>/*_AGL.h5`. Each is 1024x1024.

Pure parsing helpers (city_of, agl_path_for, group_by_city) are unit-tested;
the HuggingFace download + h5 read is I/O exercised by the smoke run.
"""
from __future__ import annotations

import collections
from pathlib import Path
from typing import Iterator

import numpy as np

_REPO = "earthflow/GAMUS"


# --------------------------- pure helpers ---------------------------

def city_of(rgb_path: str) -> str:
    """City token = characters before the first underscore of the basename."""
    name = rgb_path.split("/")[-1]
    return name.split("_", 1)[0]


def agl_path_for(rgb_path: str) -> str:
    """Map an RGB path to its matching AGL (height) path."""
    return rgb_path.replace("images/", "heights/").replace("_RGB.h5", "_AGL.h5")


def tile_id_of(rgb_path: str) -> str:
    """Stable identifier for a tile (basename without the _RGB.h5 suffix)."""
    return rgb_path.split("/")[-1].replace("_RGB.h5", "")


def group_by_city(rgb_files: list[str]) -> dict[str, int]:
    """Count tiles per city."""
    counts: dict[str, int] = collections.Counter()
    for f in rgb_files:
        counts[city_of(f)] += 1
    return dict(counts)


# --------------------------- I/O ---------------------------

def _scan_local_pairs(split: str, gamus_root: Path | None, cache_dir: Path | None) -> list[str]:
    pairs = set()
    dirs = [d for d in (gamus_root, cache_dir) if d is not None and d.is_dir()]
    for d in dirs:
        img_dir = d / "images" / split
        if not img_dir.is_dir():
            continue
        for rgb_p in img_dir.glob("*_RGB.h5"):
            rel_rgb = f"images/{split}/{rgb_p.name}"
            rel_agl = agl_path_for(rel_rgb)
            agl_found = any((target_d / rel_agl).is_file() for target_d in dirs)
            if agl_found:
                pairs.add(rel_rgb)
    return sorted(pairs)


def list_eval_tiles(
    split: str = "test",
    gamus_root: Path | str | None = None,
    offline: bool = False,
    cache_dir: Path | str | None = None,
) -> list[str]:
    """Sorted, deterministic list of RGB tile paths for a split (local-first / offline-safe)."""
    if gamus_root is not None:
        gamus_root = Path(gamus_root)
    else:
        default_root = Path(__file__).resolve().parent.parent.parent / "GAMUS"
        gamus_root = default_root if default_root.exists() else None

    cache_path = (
        Path(cache_dir) if cache_dir is not None
        else (Path(__file__).resolve().parent.parent / "data" / "gamus_cache")
    )

    local_pairs = _scan_local_pairs(split, gamus_root, cache_path)
    if local_pairs:
        return local_pairs

    if split == "val":
        train_pairs = _scan_local_pairs("train", gamus_root, cache_path)
        if train_pairs:
            val_count = min(40, max(1, len(train_pairs) // 10))
            return train_pairs[-val_count:]

    if offline:
        return []

    try:
        from huggingface_hub import list_repo_files
        all_files = list_repo_files(_REPO, repo_type="dataset")
        prefix = f"images/{split}/"
        return sorted(f for f in all_files if f.startswith(prefix) and f.endswith("_RGB.h5"))
    except Exception as e:
        print(f"GAMUS eval loader: Hub listing failed ({e}); no tiles found.")
        return []


def load_tile(
    rgb_rel: str,
    cache_dir: Path,
    gamus_root: Path | str | None = None,
    offline: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Load (from disk or cached download) and read one tile.

    Returns (rgb_uint8 HxWx3, agl_meters HxW float32). Raises on failure so the
    runner can count and skip.
    """
    import h5py

    if gamus_root is not None:
        gamus_root = Path(gamus_root)
    else:
        default_root = Path(__file__).resolve().parent.parent.parent / "GAMUS"
        gamus_root = default_root if default_root.exists() else None

    agl_rel = agl_path_for(rgb_rel)

    # Check gamus_root
    if gamus_root is not None:
        gr = Path(gamus_root)
        if (gr / rgb_rel).is_file() and (gr / agl_rel).is_file():
            with h5py.File(gr / rgb_rel, "r") as f:
                rgb = np.array(f["image"])
            with h5py.File(gr / agl_rel, "r") as f:
                agl = np.array(f["image"]).astype(np.float32)
            return _format_tile(rgb, agl)

    # Check cache_dir
    cache_path = Path(cache_dir)
    if (cache_path / rgb_rel).is_file() and (cache_path / agl_rel).is_file():
        with h5py.File(cache_path / rgb_rel, "r") as f:
            rgb = np.array(f["image"])
        with h5py.File(cache_path / agl_rel, "r") as f:
            agl = np.array(f["image"]).astype(np.float32)
        return _format_tile(rgb, agl)

    # Also check upgrade/data/gamus_cache
    gamus_cache_path = Path(__file__).resolve().parent.parent / "data" / "gamus_cache"
    if (gamus_cache_path / rgb_rel).is_file() and (gamus_cache_path / agl_rel).is_file():
        with h5py.File(gamus_cache_path / rgb_rel, "r") as f:
            rgb = np.array(f["image"])
        with h5py.File(gamus_cache_path / agl_rel, "r") as f:
            agl = np.array(f["image"]).astype(np.float32)
        return _format_tile(rgb, agl)

    if offline:
        raise FileNotFoundError(
            f"GAMUS offline mode: tile {rgb_rel} not found on disk at {gamus_root} or {cache_dir}"
        )

    # Online download via Hugging Face Hub
    from huggingface_hub import hf_hub_download

    cache_path.mkdir(parents=True, exist_ok=True)
    rgb_path = hf_hub_download(_REPO, rgb_rel, repo_type="dataset", local_dir=cache_path)
    agl_path = hf_hub_download(_REPO, agl_rel, repo_type="dataset", local_dir=cache_path)

    with h5py.File(rgb_path, "r") as f:
        rgb = np.array(f["image"])
    with h5py.File(agl_path, "r") as f:
        agl = np.array(f["image"]).astype(np.float32)

    return _format_tile(rgb, agl)


def _format_tile(rgb: np.ndarray, agl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=-1)
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb, agl


def iter_eval_tiles(
    split: str = "test",
    limit: int | None = None,
    cache_dir: Path | None = None,
    gamus_root: Path | str | None = None,
    offline: bool = False,
) -> Iterator[tuple[str, str, np.ndarray, np.ndarray]]:
    """Yield (tile_id, city, rgb, agl_meters) for each tile in the split.

    limit caps the number of tiles (for smoke tests); None = full split.
    Tiles that fail to download/read are skipped with a printed warning.
    """
    cache_dir = cache_dir or (Path(__file__).resolve().parent.parent / "data" / "gamus_eval")
    rgb_files = list_eval_tiles(split, gamus_root=gamus_root, offline=offline, cache_dir=cache_dir)
    if limit is not None:
        rgb_files = rgb_files[:limit]

    for rgb_rel in rgb_files:
        try:
            rgb, agl = load_tile(rgb_rel, cache_dir, gamus_root=gamus_root, offline=offline)
        except Exception as e:  # noqa: BLE001
            print(f"  skip {tile_id_of(rgb_rel)}: {type(e).__name__}: {e}")
            continue
        yield tile_id_of(rgb_rel), city_of(rgb_rel), rgb, agl
