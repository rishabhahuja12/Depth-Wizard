"""
MixedMetricDataset — blend GAMUS with harmonized auxiliary height sources.

Sequencing (research §2.8/§2.9): train on GAMUS ALONE first; only after the
metric model works do you blend in a small Open-Canopy (forested) slice + GBH
(global/sparse) — all through ONE height head, harmonized to a common GSD and
meters-above-ground, with balanced batches so the big set doesn't drown the small.

A "source" is any object exposing:  .name (str), .gsd (float), __len__(),
raw(i) -> (rgb_uint8 HxWx3, height_meters HxW). GAMUS and GeoTiffHeightSource
(below) both satisfy it. The mixing/harmonization is unit-tested with fakes;
real Open-Canopy/GBH ingestion is the GeoTiffHeightSource adapter (rasterio I/O).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import harmonize as hz


class MixedMetricDataset:
    """Deterministic (zero-augmentation) blend of height sources with balanced
    per-source sampling. Each item is harmonized to `target_gsd`, forced to
    meters-above-ground, and conformed to `crop`."""

    def __init__(self, sources: list, weights: dict, target_gsd: float, crop: int,
                 total_per_epoch: int):
        self.sources = sources
        self.target_gsd = target_gsd
        self.crop = crop
        sizes = {s.name: len(s) for s in sources}
        self._plan = hz.balanced_source_plan(sizes, weights, total_per_epoch)

        # Flatten into a deterministic sample list: for each source pick `count`
        # item indices evenly spaced across it (no randomness).
        self.samples: list[tuple[int, int]] = []
        for si, s in enumerate(sources):
            count = self._plan[s.name]
            if count <= 0 or len(s) == 0:
                continue
            picks = np.linspace(0, len(s) - 1, count).round().astype(int)
            self.samples.extend((si, int(i)) for i in picks)

    def source_counts(self) -> dict:
        return dict(self._plan)

    def __len__(self) -> int:
        return len(self.samples)

    def _conform(self, arr: np.ndarray, is_rgb: bool) -> np.ndarray:
        """Resize/crop a resampled array to exactly crop x crop (deterministic)."""
        from PIL import Image
        c = self.crop
        if arr.shape[0] == c and arr.shape[1] == c:
            return arr
        if is_rgb:
            return np.array(Image.fromarray(arr.astype(np.uint8)).resize((c, c), Image.BILINEAR))
        return np.array(Image.fromarray(arr.astype(np.float32), mode="F").resize((c, c), Image.BILINEAR))

    def __getitem__(self, idx: int):
        import torch
        import metric_dataset as md

        si, i = self.samples[idx]
        src = self.sources[si]
        rgb, height = src.raw(i)

        # Harmonize: resample to the common grid, force meters-above-ground.
        rgb = hz.resample_to_gsd(rgb, src.gsd, self.target_gsd, order=1)
        height = hz.resample_to_gsd(height, src.gsd, self.target_gsd, order=1)
        height = hz.conform_meters_above_ground(height)

        rgb = self._conform(rgb, is_rgb=True)
        height = self._conform(height, is_rgb=False)

        rgb_t = torch.from_numpy(md.normalize_rgb(rgb))
        depth_t = torch.from_numpy(height.copy())  # METERS
        return rgb_t, depth_t


class GeoTiffHeightSource:
    """Adapter for any paired-raster dataset (Open-Canopy, GBH, DFC2019): a folder
    of RGB GeoTIFFs and a folder of matching height GeoTIFFs at a known GSD.

    Point it at the downloaded tiles:
        GeoTiffHeightSource(name="canopy", gsd=1.5,
                            rgb_dir="data/open_canopy/rgb",
                            height_dir="data/open_canopy/chm")

    Matching is by shared stem. Height GeoTIFFs are read as meters (canopy height
    / nDSM are already above-ground). rasterio is imported lazily so importing this
    module needs no GIS stack.
    """

    def __init__(self, name: str, gsd: float, rgb_dir, height_dir,
                 rgb_glob: str = "*.tif", height_suffix: str | None = None):
        self.name = name
        self.gsd = gsd
        self.rgb_dir = Path(rgb_dir)
        self.height_dir = Path(height_dir)
        self.pairs: list[tuple[Path, Path]] = []
        for rgb_p in sorted(self.rgb_dir.glob(rgb_glob)):
            stem = rgb_p.stem
            cand = (self.height_dir / f"{stem}{height_suffix}.tif") if height_suffix \
                else (self.height_dir / f"{stem}.tif")
            if cand.exists():
                self.pairs.append((rgb_p, cand))

    def __len__(self) -> int:
        return len(self.pairs)

    def raw(self, i: int):
        import rasterio
        rgb_p, h_p = self.pairs[i]
        with rasterio.open(rgb_p) as ds:
            bands = min(ds.count, 3)
            rgb = np.stack([ds.read(b + 1) for b in range(bands)], axis=-1)
            if bands == 1:
                rgb = np.repeat(rgb[..., :1], 3, axis=-1)
        if rgb.dtype != np.uint8:  # percentile-stretch non-8-bit to viewable RGB
            out = np.empty(rgb.shape[:2] + (3,), np.uint8)
            for c in range(3):
                b = rgb[..., c].astype(np.float32)
                lo, hi = np.percentile(b[np.isfinite(b)], [2, 98]) if np.any(np.isfinite(b)) else (0, 1)
                out[..., c] = np.clip((b - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32).__mul__(255).astype(np.uint8)
            rgb = out
        with rasterio.open(h_p) as ds:
            height = ds.read(1).astype(np.float32)
        return rgb[..., :3], height
