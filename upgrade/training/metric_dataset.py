"""
P1 metric GAMUS dataset — yields targets in METERS (no [0,1] normalization).

This is the deliberate contrast with backend/training/train_gamus_full.py, whose
FullGAMUSDataset percentile-normalizes the target to [0,1] (line ~187), destroying
metric scale before the loss ever sees it. Here the nDSM stays in meters.

Augmentation is appearance-only + flips/rotations (which preserve the
height<->image correspondence). NO tilt/perspective warps — those would corrupt
the nDSM label (see research §3.3).

Pure helpers (prepare_target, random_crop_box, apply_augment) are unit-tested;
the HuggingFace download + h5 read is I/O exercised on the workstation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------- pure helpers ---------------------------

def prepare_target(agl_patch: np.ndarray) -> np.ndarray:
    """Clean an nDSM patch, KEEPING METERS: NaN->0, clamp negatives (noise/water)
    to 0 (ground). No normalization — the whole point of P1."""
    out = np.nan_to_num(agl_patch.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return np.maximum(out, 0.0)


def random_crop_box(h: int, w: int, crop: int, rng: np.random.Generator) -> tuple[int, int, int, int]:
    """Random crop box (r1, r2, c1, c2). If crop exceeds the image, use the full image."""
    if crop >= h or crop >= w:
        return 0, h, 0, w
    r1 = int(rng.integers(0, h - crop + 1))
    c1 = int(rng.integers(0, w - crop + 1))
    return r1, r1 + crop, c1, c1 + crop


def apply_augment(rgb: np.ndarray, depth: np.ndarray, hflip: bool, vflip: bool, k_rot: int):
    """Apply the SAME flips/rotation to rgb and depth (correspondence-preserving)."""
    if hflip:
        rgb, depth = np.fliplr(rgb), np.fliplr(depth)
    if vflip:
        rgb, depth = np.flipud(rgb), np.flipud(depth)
    if k_rot % 4:
        rgb, depth = np.rot90(rgb, k_rot), np.rot90(depth, k_rot)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(depth)


def normalize_rgb(rgb_patch: np.ndarray) -> np.ndarray:
    """uint8 HxWx3 -> float32 CxHxW, ImageNet-normalized (what DINOv2/DA2 expects)."""
    x = rgb_patch.astype(np.float32) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.transpose(x, (2, 0, 1)).copy()


# --------------------------- offline manifest readers ---------------------------

def offline_tiles(manifest_path) -> list[dict]:
    """Read a prefetch manifest (produced by gamus_prefetch) — list of tile dicts
    with rgb_local/agl_local/city/tile_id. Pure JSON, no network."""
    import json
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))["tiles"]


def _read_h5_pair(rgb_local: str, agl_local: str):
    import h5py
    with h5py.File(rgb_local, "r") as f:
        rgb_full = np.array(f["image"])
    with h5py.File(agl_local, "r") as f:
        agl_full = np.array(f["image"]).astype(np.float32)
    if rgb_full.ndim == 2:
        rgb_full = np.repeat(rgb_full[..., None], 3, axis=-1)
    return rgb_full[..., :3], agl_full


def iter_manifest_full_tiles(manifest_path, limit: int | None = None):
    """Yield (tile_id, city, rgb_full, agl_full) from a prefetched split, offline.
    Used for held-out evaluation without touching HuggingFace."""
    tiles = offline_tiles(manifest_path)
    if limit is not None:
        tiles = tiles[:limit]
    for t in tiles:
        try:
            rgb_full, agl_full = _read_h5_pair(t["rgb_local"], t["agl_local"])
        except Exception as e:  # noqa: BLE001
            print(f"  skip {t.get('tile_id')}: {type(e).__name__}: {e}")
            continue
        yield t["tile_id"], t["city"], rgb_full, agl_full


# --------------------------- dataset ---------------------------

class MetricGAMUSDataset:
    """torch Dataset over GAMUS RGB->nDSM(meters) pairs for metric fine-tuning.

    Two modes:
      * offline=True  -> reads a prefetch manifest and local h5 files, NO network
        (use after running gamus_prefetch; robust to internet drops).
      * offline=False -> lazily lists + downloads tiles from HuggingFace, with
        retries; caches to disk as it goes.

    Heavy imports are deferred so the pure helpers above stay unit-testable.
    """

    def __init__(self, split: str = "train", crop: int = 518, augment: bool = True,
                 cache_dir: Path | None = None, seed: int = 42, repo: str = "earthflow/GAMUS",
                 offline: bool = False, manifest_path: Path | None = None, retries: int = 4):
        from torch.utils.data import Dataset  # noqa: F401  (marker that torch is available)

        self.split = split
        self.crop = crop
        self.augment = augment
        self.repo = repo
        self.offline = offline
        self.retries = retries
        self.cache_dir = cache_dir or (Path(__file__).resolve().parent.parent / "data" / "gamus_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)

        if offline:
            mp = Path(manifest_path) if manifest_path else self.cache_dir / f"manifest_{split}.json"
            if not mp.exists():
                raise FileNotFoundError(
                    f"Offline mode but no manifest at {mp}. Run gamus_prefetch first.")
            self.tiles = offline_tiles(mp)
        else:
            from huggingface_hub import list_repo_files
            all_files = list_repo_files(repo, repo_type="dataset")
            prefix = f"images/{split}/"
            self.rgb_files = sorted(f for f in all_files if f.startswith(prefix) and f.endswith("_RGB.h5"))

    def __len__(self) -> int:
        return len(self.tiles) if self.offline else len(self.rgb_files)

    def _agl_for(self, rgb_rel: str) -> str:
        return rgb_rel.replace("images/", "heights/").replace("_RGB.h5", "_AGL.h5")

    def _load_full(self, idx: int):
        if self.offline:
            t = self.tiles[idx]
            return _read_h5_pair(t["rgb_local"], t["agl_local"])
        # online: download-with-retry, then read
        import time
        from huggingface_hub import hf_hub_download
        rgb_rel = self.rgb_files[idx]
        agl_rel = self._agl_for(rgb_rel)
        last = None
        for attempt in range(self.retries):
            try:
                rgb_p = hf_hub_download(self.repo, rgb_rel, repo_type="dataset", local_dir=self.cache_dir)
                agl_p = hf_hub_download(self.repo, agl_rel, repo_type="dataset", local_dir=self.cache_dir)
                return _read_h5_pair(rgb_p, agl_p)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"failed to load tile {idx} after {self.retries} tries: {last}")

    def __getitem__(self, idx: int):
        import torch

        rgb_full, agl_full = self._load_full(idx)

        h, w = agl_full.shape[:2]
        r1, r2, c1, c2 = random_crop_box(h, w, self.crop, self.rng)
        rgb = rgb_full[r1:r2, c1:c2]
        depth = prepare_target(agl_full[r1:r2, c1:c2])

        if self.augment:
            rgb, depth = apply_augment(
                rgb, depth,
                hflip=bool(self.rng.integers(0, 2)),
                vflip=bool(self.rng.integers(0, 2)),
                k_rot=int(self.rng.integers(0, 4)),
            )

        rgb_t = torch.from_numpy(normalize_rgb(rgb))
        depth_t = torch.from_numpy(depth.copy())  # METERS
        return rgb_t, depth_t
