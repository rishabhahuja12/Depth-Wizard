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

    def _cell(self, rgb: np.ndarray, height: np.ndarray):
        """Take a deterministic crop x crop CELL at the harmonized GSD (G3).

        The previous version *resized* the whole resampled tile to crop, which
        silently undid the resample-to-target_gsd (effective GSD became
        tile_extent/crop, not target_gsd) — so the same object appeared at very
        different pixel scales across sources. Cropping a cell instead keeps the
        true target_gsd. Only when a tile is smaller than crop do we upscale.
        ponytail: single center cell per tile (loses within-tile coverage); tile
        into a cell grid like GAMUS if that coverage ever matters.
        """
        from PIL import Image
        c = self.crop
        # keep rgb and height on the same grid (some sources ship them at diff sizes)
        if height.shape[:2] != rgb.shape[:2]:
            height = np.array(Image.fromarray(height.astype(np.float32), mode="F")
                              .resize((rgb.shape[1], rgb.shape[0]), Image.BILINEAR))
        h, w = rgb.shape[:2]
        if h >= c and w >= c:
            r0, c0 = (h - c) // 2, (w - c) // 2          # center cell, deterministic
            return rgb[r0:r0 + c, c0:c0 + c], height[r0:r0 + c, c0:c0 + c]
        rgb = np.array(Image.fromarray(rgb.astype(np.uint8)).resize((c, c), Image.BILINEAR))
        height = np.array(Image.fromarray(height.astype(np.float32), mode="F").resize((c, c), Image.BILINEAR))
        return rgb, height

    def __getitem__(self, idx: int):
        import torch
        import metric_dataset as md

        si, i = self.samples[idx]
        src = self.sources[si]
        rgb, height = src.raw(i)

        # Harmonize: resample to the common grid, force meters-above-ground, then
        # take a crop-sized cell AT that GSD (don't resize the whole tile — G3).
        rgb = hz.resample_to_gsd(rgb, src.gsd, self.target_gsd, order=1)
        height = hz.resample_to_gsd(height, src.gsd, self.target_gsd, order=1)
        height = hz.conform_meters_above_ground(height)

        rgb, height = self._cell(rgb, height)

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

    Pairing (Gap 3): the height file's stem is the RGB stem with `rgb_token`
    replaced by `height_token` (e.g. US3D `JAX_004_RGB` -> `JAX_004_AGL`) plus an
    optional `height_suffix`; with no token it's the shared stem (Open-Canopy).
    Band order (Gap 4): `rgb_bands` picks explicit 1-based bands for multi-band
    imagery (e.g. (5,3,2) for WorldView-3 8-band VNIR); default is the first 3.
    Height GeoTIFFs are read as meters. rasterio is imported lazily so importing
    this module needs no GIS stack.
    """

    def __init__(self, name: str, gsd: float, rgb_dir, height_dir,
                 rgb_glob: str = "*.tif", height_suffix: str | None = None,
                 rgb_token: str | None = None, height_token: str = "",
                 rgb_bands: tuple | None = None,
                 height_nodata: float | None = None, height_scale: float = 1.0,
                 max_height_m: float = 1000.0):
        self.name = name
        self.gsd = gsd
        self.rgb_bands = rgb_bands           # 1-based band indices; None -> first 3
        self.height_nodata = height_nodata  # explicit override; else read from raster
        self.height_scale = height_scale    # -> meters (0.1 decimeters, 0.01 cm)
        self.max_height_m = max_height_m     # post-scale sentinel/absurd cutoff -> 0
        self.rgb_dir = Path(rgb_dir)
        self.height_dir = Path(height_dir)
        # Recursive: real datasets nest by year (Open-Canopy) or AOI (US3D), so a
        # flat glob finds nothing. Match by stem via a recursive index of the height
        # dir, so nesting on either side is handled.
        rgb_files = sorted(self.rgb_dir.rglob(rgb_glob))
        self.n_rgb = len(rgb_files)          # for a loud, actionable 0-pairs message
        height_index: dict[str, Path] = {}
        for hp in self.height_dir.rglob("*.tif"):
            height_index.setdefault(hp.stem, hp)   # first-wins on duplicate stems
        self.pairs: list[tuple[Path, Path]] = []
        for rgb_p in rgb_files:
            stem = rgb_p.stem
            if rgb_token:
                stem = stem.replace(rgb_token, height_token)
            hp = height_index.get(f"{stem}{height_suffix or ''}")
            if hp is not None:
                self.pairs.append((rgb_p, hp))

    def __len__(self) -> int:
        return len(self.pairs)

    def raw(self, i: int):
        import rasterio
        rgb_p, h_p = self.pairs[i]
        with rasterio.open(rgb_p) as ds:
            if self.rgb_bands:
                rgb = np.stack([ds.read(b) for b in self.rgb_bands], axis=-1)
            else:
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
            nod = self.height_nodata if self.height_nodata is not None else ds.nodata
        height = hz.sanitize_height(height, nodata=nod, scale=self.height_scale,
                                    max_m=self.max_height_m)
        return rgb[..., :3], height
