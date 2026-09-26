"""
P1 metric GAMUS dataset — yields targets in METERS.

GAMUS RGB + AGL tiles are downloaded from Hugging Face on first use and
stored under:

    upgrade/data/gamus_cache/

After a tile has been downloaded, training reads it directly from the local
cache and does NOT contact Hugging Face again for that tile.

This version is intentionally DataLoader-worker safe and works with
workers=0 or workers>0.
"""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np


_IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
)

_IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
)


# ============================================================
# PURE HELPERS
# ============================================================

def prepare_target(agl_patch: np.ndarray) -> np.ndarray:
    """
    Clean an nDSM patch while KEEPING METERS.

    NaN / +/-inf -> 0
    Negative heights -> 0
    No normalization.
    """
    out = np.nan_to_num(
        agl_patch.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return np.maximum(out, 0.0)


def tile_grid(
    h: int,
    w: int,
    crop: int,
) -> list[tuple[int, int]]:
    """
    Deterministic non-overlapping crop origins.

    For a 1024x1024 tile and crop=504 this produces:

        2 rows x 2 cols = 4 cells

    Partial remainder regions are dropped.
    """
    if crop >= h or crop >= w:
        return [(0, 0)]

    rows = h // crop
    cols = w // crop

    return [
        (r * crop, c * crop)
        for r in range(rows)
        for c in range(cols)
    ]


def random_crop_box(
    h: int,
    w: int,
    crop: int,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    """
    Random crop helper retained for reference / optional experiments.
    """
    if crop >= h or crop >= w:
        return 0, h, 0, w

    r1 = int(rng.integers(0, h - crop + 1))
    c1 = int(rng.integers(0, w - crop + 1))

    return (
        r1,
        r1 + crop,
        c1,
        c1 + crop,
    )


def apply_augment(
    rgb: np.ndarray,
    depth: np.ndarray,
    hflip: bool,
    vflip: bool,
    k_rot: int,
):
    """
    Apply identical geometric transforms to RGB and depth.
    """
    if hflip:
        rgb = np.fliplr(rgb)
        depth = np.fliplr(depth)

    if vflip:
        rgb = np.flipud(rgb)
        depth = np.flipud(depth)

    if k_rot % 4:
        rgb = np.rot90(rgb, k_rot)
        depth = np.rot90(depth, k_rot)

    return (
        np.ascontiguousarray(rgb),
        np.ascontiguousarray(depth),
    )


def normalize_rgb(rgb_patch: np.ndarray) -> np.ndarray:
    """
    uint8 HxWx3 -> float32 CxHxW with ImageNet normalization.
    """
    x = rgb_patch.astype(np.float32) / 255.0

    x = (
        x - _IMAGENET_MEAN
    ) / _IMAGENET_STD

    return np.transpose(
        x,
        (2, 0, 1),
    ).copy()


# ============================================================
# HDF5 READER
# ============================================================

def _read_h5_pair(
    rgb_local: str,
    agl_local: str,
    box: tuple[int, int, int] | None = None,
):
    """
    Read matching RGB + AGL HDF5 files.

    If box=(r0,c0,crop), only that region is read.
    """

    import h5py

    if box is not None:

        r0, c0, crop = box

        with h5py.File(rgb_local, "r") as f:
            rgb = np.array(
                f["image"][
                    r0:r0 + crop,
                    c0:c0 + crop,
                ]
            )

        with h5py.File(agl_local, "r") as f:
            agl = np.array(
                f["image"][
                    r0:r0 + crop,
                    c0:c0 + crop,
                ]
            ).astype(np.float32)

    else:

        with h5py.File(rgb_local, "r") as f:
            rgb = np.array(f["image"])

        with h5py.File(agl_local, "r") as f:
            agl = np.array(
                f["image"]
            ).astype(np.float32)

    if rgb.ndim == 2:
        rgb = np.repeat(
            rgb[..., None],
            3,
            axis=-1,
        )

    return rgb[..., :3], agl


# ============================================================
# DATASET
# ============================================================

class MetricGAMUSDataset:
    """
    GAMUS RGB -> metric nDSM dataset.

    IMPORTANT:

    1. First access downloads missing files from Hugging Face.
    2. Files are stored locally under cache_dir.
    3. Subsequent access uses the local files directly.
    4. Hugging Face is NOT contacted when both files exist.
    """

    TILE_SIZE = 1024

    def __init__(
        self,
        split: str = "train",
        crop: int = 512,
        augment: bool = False,
        cache_dir: Path | None = None,
        seed: int = 42,
        repo: str = "earthflow/GAMUS",
        retries: int = 4,
        tile_size: int | None = None,
        gamus_root: Path | str | None = None,
        offline: bool = False,
        val_tiles: int = 40,
    ):

        self.split = split
        self.crop = crop
        self.augment = augment
        self.tile_size = tile_size or self.TILE_SIZE
        self.repo = repo
        self.retries = retries
        self.offline = offline

        if gamus_root is not None:
            self.gamus_root = Path(gamus_root)
        else:
            default_root = Path(__file__).resolve().parent.parent.parent / "GAMUS"
            self.gamus_root = default_root if default_root.exists() else None

        self.cache_dir = (
            Path(cache_dir)
            if cache_dir is not None
            else (
                Path(__file__).resolve().parent.parent
                / "data"
                / "gamus_cache"
            )
        )

        self.cache_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.rng = np.random.default_rng(seed)

        # ----------------------------------------------------
        # DISCOVER DATASET FILES (LOCAL FIRST / OFFLINE SAFE)
        # ----------------------------------------------------
        self.rgb_files = self._discover_tiles(split, val_tiles=val_tiles)
        n_tiles = len(self.rgb_files)

        if n_tiles == 0 and self.offline:
            raise FileNotFoundError(
                f"GAMUS offline mode: No paired tiles found for split '{split}' in "
                f"{self.gamus_root} or {self.cache_dir}."
            )

        # ----------------------------------------------------
        # BUILD DETERMINISTIC SAMPLE INDEX
        # ----------------------------------------------------
        cells = tile_grid(
            self.tile_size,
            self.tile_size,
            crop,
        )

        self.index = [
            (tile_i, r0, c0)
            for tile_i in range(n_tiles)
            for r0, c0 in cells
        ]

        print(
            f"GAMUS '{split}': "
            f"{len(self.index)} training samples "
            f"from {n_tiles} tiles "
            f"with crop={crop}"
            + (" (OFFLINE)" if self.offline else ""),
            flush=True,
        )

    # ========================================================
    # DISCOVERY & PATH HELPERS
    # ========================================================

    def _scan_local_pairs(self, split: str) -> list[str]:
        """Scan gamus_root and cache_dir for matching RGB and AGL pairs on disk."""
        pairs = set()
        dirs_to_check = []
        if self.gamus_root and self.gamus_root.is_dir():
            dirs_to_check.append(self.gamus_root)
        if self.cache_dir and self.cache_dir.is_dir():
            dirs_to_check.append(self.cache_dir)

        for d in dirs_to_check:
            img_dir = d / "images" / split
            if not img_dir.is_dir():
                continue
            for rgb_p in img_dir.glob("*_RGB.h5"):
                rel_rgb = f"images/{split}/{rgb_p.name}"
                rel_agl = self._agl_for(rel_rgb)
                agl_found = False
                for target_d in dirs_to_check:
                    if (target_d / rel_agl).is_file():
                        agl_found = True
                        break
                if agl_found:
                    pairs.add(rel_rgb)

        return sorted(pairs)

    def _discover_tiles(self, split: str, val_tiles: int = 40) -> list[str]:
        local_pairs = self._scan_local_pairs(split)

        if local_pairs:
            print(f"GAMUS '{split}': found {len(local_pairs)} local paired tiles on disk", flush=True)
            return local_pairs

        if split == "val":
            train_pairs = self._scan_local_pairs("train")
            if train_pairs:
                val_count = min(val_tiles, max(1, len(train_pairs)))
                val_sample = train_pairs[:val_count]
                print(f"GAMUS 'val': Using {len(val_sample)} sample tiles from local train split for evaluation tracking.", flush=True)
                return val_sample

        if self.offline:
            print(f"GAMUS '{split}': offline mode and no local paired tiles found.", flush=True)
            return []

        try:
            print(f"GAMUS: querying Hugging Face dataset file list: {self.repo}", flush=True)
            from huggingface_hub import list_repo_files
            all_files = list_repo_files(self.repo, repo_type="dataset")
            prefix = f"images/{split}/"
            rgb = sorted(f for f in all_files if f.startswith(prefix) and f.endswith("_RGB.h5"))
            print(f"GAMUS '{split}': {len(rgb)} tiles listed from HF Hub ({self.repo})", flush=True)
            return rgb
        except Exception as e:
            print(f"GAMUS '{split}': Hub query failed ({e}); no tiles found.", flush=True)
            return []

    def _agl_for(
        self,
        rgb_rel: str,
    ) -> str:
        """
        Convert:

            images/train/XXX_RGB.h5

        to:

            heights/train/XXX_AGL.h5
        """

        return (
            rgb_rel
            .replace(
                "images/",
                "heights/",
            )
            .replace(
                "_RGB.h5",
                "_AGL.h5",
            )
        )

    def _local_path(
        self,
        relative_path: str,
    ) -> Path:
        """
        Return the local path where the file is found (checking gamus_root first, then cache_dir).
        """
        if self.gamus_root is not None:
            p_root = self.gamus_root / relative_path
            if p_root.is_file():
                return p_root

        p_cache = self.cache_dir / relative_path
        if p_cache.is_file():
            return p_cache

        return (
            self.gamus_root / relative_path
            if self.gamus_root is not None
            else p_cache
        )

    # ========================================================
    # LOAD ONE TILE/CELL
    # ========================================================

    def _read_cell(
        self,
        tile_i: int,
        box: tuple[int, int, int],
    ):

        from huggingface_hub import hf_hub_download

        rgb_rel = self.rgb_files[tile_i]

        agl_rel = self._agl_for(
            rgb_rel
        )

        rgb_local = self._local_path(
            rgb_rel
        )

        agl_local = self._local_path(
            agl_rel
        )

        last_error = None

        # Check local files first
        if (
            rgb_local.is_file()
            and agl_local.is_file()
        ):
            return _read_h5_pair(
                str(rgb_local),
                str(agl_local),
                box=box,
            )

        if self.offline:
            raise FileNotFoundError(
                f"GAMUS offline mode: tile {tile_i} ({rgb_rel}) not found on disk "
                f"at {self.gamus_root} or {self.cache_dir}"
            )

        # ----------------------------------------------------
        # RETRIES (ONLINE DOWNLOAD ONLY)
        # ----------------------------------------------------
        for attempt in range(self.retries):
            try:
                # ====================================================
                # RGB DOWNLOAD
                # ====================================================
                if not rgb_local.is_file():
                    print(
                        f"GAMUS DOWNLOAD: tile={tile_i} RGB={rgb_rel}",
                        flush=True,
                    )
                    downloaded_rgb = hf_hub_download(
                        self.repo,
                        rgb_rel,
                        repo_type="dataset",
                        local_dir=self.cache_dir,
                    )
                    rgb_local = Path(downloaded_rgb)

                # ====================================================
                # AGL DOWNLOAD
                # ====================================================
                if not agl_local.is_file():
                    print(
                        f"GAMUS DOWNLOAD: tile={tile_i} AGL={agl_rel}",
                        flush=True,
                    )
                    downloaded_agl = hf_hub_download(
                        self.repo,
                        agl_rel,
                        repo_type="dataset",
                        local_dir=self.cache_dir,
                    )
                    agl_local = Path(downloaded_agl)

                # ====================================================
                # VERIFY
                # ====================================================
                if not rgb_local.is_file():
                    raise FileNotFoundError(
                        f"RGB file missing after download: {rgb_local}"
                    )
                if not agl_local.is_file():
                    raise FileNotFoundError(
                        f"AGL file missing after download: {agl_local}"
                    )

                # ====================================================
                # READ
                # ====================================================
                return _read_h5_pair(
                    str(rgb_local),
                    str(agl_local),
                    box=box,
                )

            except Exception as e:

                last_error = e

                print(
                    f"GAMUS tile {tile_i} "
                    f"load failed "
                    f"(attempt {attempt + 1}/"
                    f"{self.retries}): "
                    f"{repr(e)}",
                    flush=True,
                )

                if attempt < self.retries - 1:

                    wait = 2.0 * (
                        attempt + 1
                    )

                    print(
                        f"Retrying in "
                        f"{wait:.1f}s...",
                        flush=True,
                    )

                    time.sleep(wait)

        raise RuntimeError(
            f"Failed to load GAMUS tile "
            f"{tile_i} after "
            f"{self.retries} attempts: "
            f"{last_error}"
        )

    # ========================================================
    # SIZE SAFETY
    # ========================================================

    def _conform(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
    ):

        if (
            rgb.shape[0] == self.crop
            and rgb.shape[1] == self.crop
        ):
            return rgb, depth

        from PIL import Image

        rgb = np.array(
            Image.fromarray(
                rgb
            ).resize(
                (
                    self.crop,
                    self.crop,
                ),
                Image.BILINEAR,
            )
        )

        depth = np.array(
            Image.fromarray(
                depth,
                mode="F",
            ).resize(
                (
                    self.crop,
                    self.crop,
                ),
                Image.BILINEAR,
            )
        )

        return rgb, depth

    # ========================================================
    # DATASET API
    # ========================================================

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(
        self,
        idx: int,
    ):

        import torch

        tile_i, r0, c0 = self.index[idx]

        rgb, agl = self._read_cell(
            tile_i,
            box=(
                r0,
                c0,
                self.crop,
            ),
        )

        rgb, agl = self._conform(
            rgb,
            agl,
        )

        depth = prepare_target(
            agl
        )

        # ----------------------------------------------------
        # OPTIONAL AUGMENTATION
        # ----------------------------------------------------

        if self.augment:

            rgb, depth = apply_augment(
                rgb,
                depth,
                hflip=bool(
                    self.rng.integers(
                        0,
                        2,
                    )
                ),
                vflip=bool(
                    self.rng.integers(
                        0,
                        2,
                    )
                ),
                k_rot=int(
                    self.rng.integers(
                        0,
                        4,
                    )
                ),
            )

        # ----------------------------------------------------
        # TENSORS
        # ----------------------------------------------------

        rgb_t = torch.from_numpy(
            normalize_rgb(rgb)
        )

        depth_t = torch.from_numpy(
            depth.copy()
        )

        return (
            rgb_t,
            depth_t,
        )
