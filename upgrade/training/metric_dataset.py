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


# --------------------------- dataset ---------------------------

class MetricGAMUSDataset:
    """torch Dataset over GAMUS RGB->nDSM(meters) pairs for metric fine-tuning.

    Imports of torch / h5py / huggingface_hub are deferred to construction so the
    pure helpers above stay importable (and unit-testable) without them.
    """

    def __init__(self, split: str = "train", crop: int = 518, augment: bool = True,
                 cache_dir: Path | None = None, seed: int = 42, repo: str = "earthflow/GAMUS"):
        from torch.utils.data import Dataset  # noqa: F401  (marker that torch is available)
        from huggingface_hub import list_repo_files

        self.split = split
        self.crop = crop
        self.augment = augment
        self.repo = repo
        self.cache_dir = cache_dir or (Path(__file__).resolve().parent.parent / "data" / "gamus_train")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)

        all_files = list_repo_files(repo, repo_type="dataset")
        prefix = f"images/{split}/"
        self.rgb_files = sorted(f for f in all_files if f.startswith(prefix) and f.endswith("_RGB.h5"))

    def __len__(self) -> int:
        return len(self.rgb_files)

    def _agl_for(self, rgb_rel: str) -> str:
        return rgb_rel.replace("images/", "heights/").replace("_RGB.h5", "_AGL.h5")

    def __getitem__(self, idx: int):
        import h5py
        import torch
        from huggingface_hub import hf_hub_download

        rgb_rel = self.rgb_files[idx]
        agl_rel = self._agl_for(rgb_rel)
        rgb_path = hf_hub_download(self.repo, rgb_rel, repo_type="dataset", local_dir=self.cache_dir)
        agl_path = hf_hub_download(self.repo, agl_rel, repo_type="dataset", local_dir=self.cache_dir)

        with h5py.File(rgb_path, "r") as f:
            rgb_full = np.array(f["image"])
        with h5py.File(agl_path, "r") as f:
            agl_full = np.array(f["image"]).astype(np.float32)

        if rgb_full.ndim == 2:
            rgb_full = np.repeat(rgb_full[..., None], 3, axis=-1)
        rgb_full = rgb_full[..., :3]

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
