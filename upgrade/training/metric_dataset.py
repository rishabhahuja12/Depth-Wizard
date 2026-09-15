"""
P1 metric GAMUS dataset — yields targets in METERS (no [0,1] normalization).

This is the deliberate contrast with backend/training/train_gamus_full.py, whose
FullGAMUSDataset percentile-normalizes the target to [0,1] (line ~187), destroying
metric scale before the loss ever sees it. Here the nDSM stays in meters.

Tiles are streamed directly from the HuggingFace Hub mirror (earthflow/GAMUS) and
cached locally in cache_dir on first download — no prefetch step is needed.
Subsequent runs reuse the local cache transparently via hf_hub_download.

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


def tile_grid(h: int, w: int, crop: int) -> list[tuple[int, int]]:
    """Deterministic non-overlapping cell origins (r0, c0) covering the tile.

    Zero-augmentation training uses this instead of a random crop: every cell is
    exactly crop x crop, in a fixed row-major order, so there is no randomness.
    A partial remainder that can't fill a full cell is dropped. If the crop is
    larger than the tile, a single (0, 0) cell is returned (the caller resizes)."""
    if crop >= h or crop >= w:
        return [(0, 0)]
    rows = h // crop
    cols = w // crop
    return [(r * crop, c * crop) for r in range(rows) for c in range(cols)]


def random_crop_box(h: int, w: int, crop: int, rng: np.random.Generator) -> tuple[int, int, int, int]:
    """Random crop box (r1, r2, c1, c2). Retained for reference/opt-in only — the
    default pipeline uses deterministic tile_grid (zero augmentation)."""
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


# --------------------------- dataset ---------------------------


def _read_h5_pair(rgb_local: str, agl_local: str, box: tuple[int, int, int] | None = None):
    """Read an RGB+AGL pair. If box=(r0, c0, crop), slice that cell directly from
    the h5 (no full-tile load); else read the whole tile."""
    import h5py
    if box is not None:
        r0, c0, crop = box
        with h5py.File(rgb_local, "r") as f:
            rgb_full = np.array(f["image"][r0:r0 + crop, c0:c0 + crop])
        with h5py.File(agl_local, "r") as f:
            agl_full = np.array(f["image"][r0:r0 + crop, c0:c0 + crop]).astype(np.float32)
    else:
        with h5py.File(rgb_local, "r") as f:
            rgb_full = np.array(f["image"])
        with h5py.File(agl_local, "r") as f:
            agl_full = np.array(f["image"]).astype(np.float32)
    if rgb_full.ndim == 2:
        rgb_full = np.repeat(rgb_full[..., None], 3, axis=-1)
    return rgb_full[..., :3], agl_full


# --------------------------- dataset ---------------------------

class MetricGAMUSDataset:
    """torch Dataset over GAMUS RGB->nDSM(meters) pairs for metric fine-tuning.

    Tiles are listed from the HuggingFace Hub mirror (earthflow/GAMUS) and
    downloaded on-demand via hf_hub_download, which caches each file to
    cache_dir on first use — no prefetch step needed. Subsequent runs
    reuse cached files transparently.

    Sampling is ZERO-AUGMENTATION by default: each 1024 tile is partitioned into
    deterministic non-overlapping `crop`-sized cells (tile_grid). No random crop,
    no flips/rotations, no photometric jitter — raw tiles only. `augment=True`
    re-enables the (label-safe) flip/rotate path for opt-in experiments.

    Heavy imports are deferred so the pure helpers above stay unit-testable.
    """

    TILE_SIZE = 1024  # GAMUS tiles are uniformly 1024x1024

    def __init__(self, split: str = "train", crop: int = 512, augment: bool = False,
                 cache_dir: Path | None = None, seed: int = 42, repo: str = "earthflow/GAMUS",
                 retries: int = 4, tile_size: int | None = None):
        from torch.utils.data import Dataset  # noqa: F401  (marker that torch is available)

        self.split = split
        self.crop = crop
        self.augment = augment
        self.tile_size = tile_size or self.TILE_SIZE
        self.repo = repo
        self.retries = retries
        self.cache_dir = cache_dir or (Path(__file__).resolve().parent.parent / "data" / "gamus_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)

        # List tiles from HF Hub (result is stable / sorted, so the index is deterministic).
        from huggingface_hub import list_repo_files
        all_files = list_repo_files(repo, repo_type="dataset")
        prefix = f"images/{split}/"
        self.rgb_files = sorted(f for f in all_files if f.startswith(prefix) and f.endswith("_RGB.h5"))
        n_tiles = len(self.rgb_files)
        print(f"GAMUS '{split}': {n_tiles} tiles listed from HF Hub ({repo})")

        # Flatten (tile, cell) into a deterministic sample index.
        cells = tile_grid(self.tile_size, self.tile_size, crop)
        self.index = [(ti, r0, c0) for ti in range(n_tiles) for (r0, c0) in cells]

    def __len__(self) -> int:
        return len(self.index)

    def _agl_for(self, rgb_rel: str) -> str:
        return rgb_rel.replace("images/", "heights/").replace("_RGB.h5", "_AGL.h5")

    def _read_cell(self, tile_i: int, box):
        """Download (cached) the tile from HF Hub and slice the requested cell."""
        import time
        from huggingface_hub import hf_hub_download
        rgb_rel = self.rgb_files[tile_i]
        agl_rel = self._agl_for(rgb_rel)
        last = None
        for attempt in range(self.retries):
            try:
                rgb_p = hf_hub_download(self.repo, rgb_rel, repo_type="dataset", local_dir=self.cache_dir)
                agl_p = hf_hub_download(self.repo, agl_rel, repo_type="dataset", local_dir=self.cache_dir)
                return _read_h5_pair(rgb_p, agl_p, box=box)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"failed to load tile {tile_i} after {self.retries} tries: {last}")

    def _conform(self, rgb: np.ndarray, depth: np.ndarray):
        """Guarantee crop x crop (safety net for any tile not exactly TILE_SIZE)."""
        if rgb.shape[0] == self.crop and rgb.shape[1] == self.crop:
            return rgb, depth
        from PIL import Image
        rgb = np.array(Image.fromarray(rgb).resize((self.crop, self.crop), Image.BILINEAR))
        depth = np.array(Image.fromarray(depth, mode="F").resize((self.crop, self.crop), Image.BILINEAR))
        return rgb, depth

    def __getitem__(self, idx: int):
        import torch

        tile_i, r0, c0 = self.index[idx]
        rgb, agl = self._read_cell(tile_i, box=(r0, c0, self.crop))
        rgb, agl = self._conform(rgb, agl)
        depth = prepare_target(agl)

        if self.augment:  # opt-in only; default pipeline is zero-augmentation
            rgb, depth = apply_augment(
                rgb, depth,
                hflip=bool(self.rng.integers(0, 2)),
                vflip=bool(self.rng.integers(0, 2)),
                k_rot=int(self.rng.integers(0, 4)),
            )

        rgb_t = torch.from_numpy(normalize_rgb(rgb))
        depth_t = torch.from_numpy(depth.copy())  # METERS
        return rgb_t, depth_t
