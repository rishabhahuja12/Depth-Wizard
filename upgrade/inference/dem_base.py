"""
P1b — Absolute DSM via a real DEM base terrain (research §2A).

The metric model predicts nDSM (height above LOCAL ground) — no sea-level datum,
no natural slope. To make elevations absolute and the flood sim physical:

    DSM_absolute(x,y) = DTM_base(x,y) + nDSM_pred(x,y)

DTM_base comes from a real coarse DEM (Copernicus GLO-30 / SRTM; CartoDEM for
India), fetched for the tile footprint and upsampled to the grid. This REPURPOSES
calibration.py: it stops inventing ground (morphological min-filter) and scale
(percentile α), and instead composes a real base with the AI's metric nDSM.

Composition + upsampling are pure/tested. The DEM fetch is an adapter: prefer a
bundled/cached local DEM (offline demo) over a network call.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def compose_absolute_dsm(ndsm: np.ndarray, dtm_base: np.ndarray) -> np.ndarray:
    """DSM_absolute = DTM_base + nDSM. dtm_base is resized to nDSM's grid if needed."""
    if dtm_base.shape != ndsm.shape:
        dtm_base = upsample_dem_to_grid(dtm_base, ndsm.shape)
    return (dtm_base.astype(np.float32) + ndsm.astype(np.float32))


def upsample_dem_to_grid(dem_coarse: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Bilinearly resample a coarse DEM to the fine (H, W) tile grid."""
    from scipy.ndimage import zoom
    h, w = out_hw
    fh, fw = h / dem_coarse.shape[0], w / dem_coarse.shape[1]
    return zoom(dem_coarse.astype(np.float32), (fh, fw), order=1)


def flat_base(shape: tuple[int, int], value: float = 0.0) -> np.ndarray:
    """Flat base terrain for non-georeferenced inputs (no datum recoverable)."""
    return np.full(shape, value, dtype=np.float32)


def absolute_from_ndsm(ndsm: np.ndarray, dem_coarse: np.ndarray | None) -> np.ndarray:
    """Compose absolute elevation. If dem_coarse is None (non-georef), use a flat
    base so the result equals the nDSM (relative mode)."""
    base = flat_base(ndsm.shape) if dem_coarse is None else upsample_dem_to_grid(dem_coarse, ndsm.shape)
    return compose_absolute_dsm(ndsm, base)


# --------------------------- DEM fetch adapter (offline-first) ---------------------------

def load_local_dem(dem_path, bounds: dict | None = None) -> np.ndarray:
    """Read a bundled/cached DEM GeoTIFF (optionally windowed to `bounds` in the
    DEM's CRS). Bundle a DEM for demo AOIs so the jury run never hits the network."""
    import rasterio
    from rasterio.windows import from_bounds
    with rasterio.open(dem_path) as ds:
        if bounds:
            win = from_bounds(bounds["left"], bounds["bottom"],
                              bounds["right"], bounds["top"], ds.transform)
            return ds.read(1, window=win).astype(np.float32)
        return ds.read(1).astype(np.float32)


def dem_for_tile(bounds: dict, crs: str, cache_dir: Path | None = None,
                 bundled_dem: Path | None = None) -> np.ndarray | None:
    """Return a coarse DEM for a georeferenced tile, offline-first:
      1. a bundled DEM covering the AOI (preferred — offline, jury-safe), else
      2. None (caller falls back to flat base).
    A live Copernicus/SRTM fetch can be added here, but is intentionally NOT the
    default so demo runs never depend on the network."""
    if bundled_dem and Path(bundled_dem).exists():
        try:
            return load_local_dem(bundled_dem, bounds)
        except Exception:  # noqa: BLE001
            return None
    return None
