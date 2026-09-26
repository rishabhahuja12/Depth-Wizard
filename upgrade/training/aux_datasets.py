"""
Auxiliary height datasets — all paired RGB + height-in-meters, all plugging into the
tested GeoTiffHeightSource and mixing through MixedMetricDataset onto the SAME height
head (research §2.8/§2.9). Sequencing: blend these ONLY after GAMUS-alone beats the
baseline. Adopted flow: GAMUS → +Open-Canopy → +M4Heights → +DFC2023 → NL builder;
US3D/GeoNRW are fallbacks. See upgrade/docs/DATASETS_GUIDE.md.

Confirmed specs (see DATASETS_GUIDE / DATASET_LICENSES):
  * Open-Canopy — HF AI4Forest/Open-Canopy, 1.5 m, France; SPOT RGB + LiDAR canopy
    height (m). Ships as per-year VRTs + geometries.geojson — pre-tile before use.
  * M4Heights — HF Rituxx96x/M4Heights (GATED), 1 m aerial RGB + building height (m),
    NL/DE/CH. Zipped + Sentinel-heavy — extract the aerial slice before use.
  * DFC2023 T2 — IEEE DataPort (registration), 0.5-1 m optical + ~2 m nDSM (m), global.
  * US3D/DFC2019 — IEEE DataPort, ~0.3 m 3-band pansharpened RGB + AGL (m), US. Naming:
    RGB is per-view (JAX_004_006_RGB), AGL per-tile (JAX_004_AGL) — see us3d_source.
  * GBH — DROPPED (paired RGB not public). GeoNRW — on hold (ships DSM, needs DTM).

REAL-WORLD CAVEAT: the exact on-disk folder names must be confirmed against the
actual download — they are parameters here (rgb_subdir / height_subdir), not
hard-coded, precisely so you can point them at the real layout on the workstation.
"""
from __future__ import annotations

from pathlib import Path

from multi_dataset import GeoTiffHeightSource

# Verified GSDs (meters/pixel).
OPEN_CANOPY_GSD = 1.5
GBH_GSD = 3.0
GEONRW_GSD = 1.0
M4HEIGHTS_GSD = 1.0
DFC2023_GSD = 0.5      # optical GSD; the ~2 m nDSM is aligned to it by the reader
US3D_GSD = 0.3         # WorldView-3 pansharpened RGB


def open_canopy_source(root, rgb_subdir: str = "images", height_subdir: str = "canopy_height",
                       rgb_glob: str = "*.tif", **kw) -> GeoTiffHeightSource:
    """Open-Canopy as a height source (canopy height is already meters-above-ground).

    Supports either:
    1. Standard layout: images/ (*.tif) and canopy_height/ (*.tif) with shared stems
    2. Raw HF repo layout: canopy_height/<year>/spot and canopy_height/<year>/lidar
    """
    root = Path(root)
    if not (root / rgb_subdir).exists() and (root / "canopy_height").exists():
        spot_files = list((root / "canopy_height").rglob("compressed_pansharpened_*.tif"))
        if spot_files and "rgb_token" not in kw:
            rgb_subdir = "canopy_height"
            height_subdir = "canopy_height"
            kw.setdefault("rgb_glob", "compressed_pansharpened_*.tif")
            kw.setdefault("rgb_token", "compressed_pansharpened_")
            kw.setdefault("height_token", "compressed_lidar_")
    return GeoTiffHeightSource(name="open_canopy", gsd=OPEN_CANOPY_GSD,
                               rgb_dir=root / rgb_subdir, height_dir=root / height_subdir,
                               rgb_glob=rgb_glob, **kw)


def gbh_source(root, rgb_subdir: str = "rgb", height_subdir: str = "ndsm",
               rgb_glob: str = "*.tif", **kw) -> GeoTiffHeightSource:
    """GBH as a height source (nDSM is meters-above-ground). DROPPED from the plan.

    Verified: the paired (RGB + nDSM) GBH training data is NOT publicly available —
    the imagery is PLANET PlanetScope (proprietary); only the derived global height
    product (GBA.Height, heights only) is public (mediaTUM), and HF has only
    polygons/LoD1. So GBH cannot feed our RGB->height task; use Open-Canopy instead.
    This adapter remains only for anyone who separately holds licensed paired
    rasters. See upgrade/docs/DATASET_LICENSES.md.


    Data (GBA.Height) is on mediaTUM (NOT HuggingFace); the code repo
    (github.com/zhu-xlab/GlobalBuildingAtlas) is code only. License is CONFIRMED
    CC BY-NC 4.0 — "you may not use the material for commercial purposes" — with
    attribution + citation required (see upgrade/docs/DATASET_LICENSES.md). Use ONLY if
    the whole deliverable is non-commercial; a model trained on NC data may inherit
    the restriction (contested). Otherwise blend Open-Canopy alone (omit gbh_root).
    Adjust subdirs to the real mediaTUM layout."""
    root = Path(root)
    return GeoTiffHeightSource(name="gbh", gsd=GBH_GSD,
                               rgb_dir=root / rgb_subdir, height_dir=root / height_subdir,
                               rgb_glob=rgb_glob, **kw)


def geonrw_source(root, rgb_subdir: str = "rgb", height_subdir: str = "ndsm",
                  rgb_glob: str = "*.tif", **kw) -> GeoTiffHeightSource:
    """GeoNRW as a height source. ⚠ ON HOLD — needs a prep step the others don't.

    GeoNRW ships aerial RGB + a LiDAR ELEVATION raster (DSM/DEM, first-return
    surface — the HF mirror torchgeo/geonrw), NOT above-ground height. Point this
    at a `height_subdir` of DERIVED nDSM, produced offline by `derive_ndsm()` from
    the DSM plus a matching bare-earth DTM (NRW 'DGM' open-data — the HF mirror does
    NOT include it). License: Data licence Germany attribution 2.0 (commercial-OK).
    Adjust subdirs to the real download layout."""
    root = Path(root)
    return GeoTiffHeightSource(name="geonrw", gsd=GEONRW_GSD,
                               rgb_dir=root / rgb_subdir, height_dir=root / height_subdir,
                               rgb_glob=rgb_glob, **kw)


def m4heights_source(root, rgb_subdir: str = "rgb", height_subdir: str = "height",
                     rgb_glob: str = "*.tif", **kw) -> GeoTiffHeightSource:
    """M4Heights aerial slice as a height source (building height, meters-AGL).

    ⚠ The HF repo (Rituxx96x/M4Heights) is GATED, zipped, and Sentinel-heavy. Extract
    only the 1 m AERIAL ortho + building-height rasters into rgb_subdir/height_subdir
    (matched stems) before pointing this at them — see prefetch_m4heights()."""
    root = Path(root)
    return GeoTiffHeightSource(name="m4heights", gsd=M4HEIGHTS_GSD,
                               rgb_dir=root / rgb_subdir, height_dir=root / height_subdir,
                               rgb_glob=rgb_glob, **kw)


def dfc2023_source(root, rgb_subdir: str = "rgb", height_subdir: str = "ndsm",
                   rgb_glob: str = "*.tif", **kw) -> GeoTiffHeightSource:
    """DFC2023 Track 2 as a height source (nDSM, meters-AGL), global diversity.

    Use the TRAIN split (val/test GT withheld); drop the SAR channel. nodata in the
    nDSM is cleaned by the reader (sanitize_height). Some tiles are known to be
    slightly misaligned — down-weight via --aux-weights if it hurts."""
    root = Path(root)
    return GeoTiffHeightSource(name="dfc2023", gsd=DFC2023_GSD,
                               rgb_dir=root / rgb_subdir, height_dir=root / height_subdir,
                               rgb_glob=rgb_glob, **kw)


def us3d_source(root, rgb_subdir: str = "images", height_subdir: str = "truth",
                rgb_glob: str = "*_RGB.tif", rgb_token: str = "RGB",
                height_token: str = "AGL") -> GeoTiffHeightSource:
    """US3D/DFC2019 as a height source (AGL, meters). ⚠ FALLBACK, naming caveat.

    RGB is 3-band pansharpened (no band-select needed). Filenames: RGB is per-view
    (JAX_004_006_RGB.tif) but AGL is per-tile (JAX_004_AGL.tif) — so the simple
    RGB->AGL token remap only pairs when the download is one-view-per-tile (or you've
    renamed views to match). Many-views-per-tile needs a custom stem rule; confirm
    against the real download."""
    root = Path(root)
    return GeoTiffHeightSource(name="us3d", gsd=US3D_GSD,
                               rgb_dir=root / rgb_subdir, height_dir=root / height_subdir,
                               rgb_glob=rgb_glob, rgb_token=rgb_token, height_token=height_token)


def split_aux_train_val(sources, val_every: int = 10):
    """Split each aux source into (train, val) with NO overlap, so held-out aux
    tiles can validate the very landscapes the blend adds (G2). Deterministic:
    every `val_every`-th paired tile goes to val, the rest to train. Returns
    (train_sources, val_sources) as shallow copies with sliced `.pairs`."""
    import copy
    train, val = [], []
    for s in sources:
        n = len(s)
        if n == 0:
            continue
        val_idx = set(range(0, n, val_every)) if n >= val_every else {n - 1}
        if len(val_idx) >= n:            # never empty the train side
            val_idx = {n - 1}
        tr = copy.copy(s); tr.pairs = [p for i, p in enumerate(s.pairs) if i not in val_idx]
        va = copy.copy(s); va.pairs = [p for i, p in enumerate(s.pairs) if i in val_idx]
        if tr.pairs:
            train.append(tr)
        if va.pairs:
            val.append(va)
    return train, val


def build_aux_sources(oc_root=None, gbh_root=None, geonrw_root=None,
                      m4h_root=None, dfc_root=None, us3d_root=None, **kw) -> list:
    """Assemble whichever auxiliary sources have a root provided and actually
    contain paired tiles. Skips missing/empty roots but logs each one, so a
    mistyped root doesn't silently vanish from the blend."""
    sources = []
    for root, factory, prefix in ((oc_root, open_canopy_source, "oc_"),
                                  (m4h_root, m4heights_source, "m4h_"),
                                  (dfc_root, dfc2023_source, "dfc_"),
                                  (us3d_root, us3d_source, "us3d_"),
                                  (gbh_root, gbh_source, "gbh_"),
                                  (geonrw_root, geonrw_source, "geonrw_")):
        if not root:
            continue
        s = factory(root, **{k[len(prefix):]: v for k, v in kw.items() if k.startswith(prefix)})
        n = len(s)
        if n > 0:
            print(f"aux source '{s.name}' at {root}: {n} paired tiles")
            sources.append(s)
        elif getattr(s, "n_rgb", 0) > 0:
            print(f"aux source '{s.name}' at {root}: 0 pairs from {s.n_rgb} RGB tiles "
                  f"— names didn't match a height file (check height_subdir / rgb_token/height_token)")
        else:
            print(f"aux source '{s.name}' at {root}: EMPTY — no RGB tiles found "
                  f"(check root / rgb_subdir / rgb_glob / download)")
    return sources


def derive_ndsm(dsm_dir, dtm_dir, out_dir, glob: str = "*.tif") -> int:
    """Write nDSM GeoTIFFs = DSM - DTM (floored at 0) for GeoNRW (⚠ on hold).

    Matches DSM and DTM by shared stem, preserves the DSM's georeferencing, and
    writes to `out_dir` (the `height_subdir` geonrw_source reads). Returns the
    count written. Requires rasterio (imported lazily)."""
    import rasterio
    import harmonize as hz
    dsm_dir, dtm_dir, out_dir = Path(dsm_dir), Path(dtm_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for dsm_p in sorted(dsm_dir.glob(glob)):
        dtm_p = dtm_dir / dsm_p.name
        if not dtm_p.exists():
            print(f"  no DTM for {dsm_p.name} — skipped")
            continue
        with rasterio.open(dsm_p) as d:
            dsm, profile = d.read(1).astype("float32"), d.profile
        with rasterio.open(dtm_p) as t:
            dtm = t.read(1).astype("float32")
        ndsm = hz.ndsm_from_dsm_dtm(dsm, dtm)
        profile.update(dtype="float32", count=1, nodata=None)
        with rasterio.open(out_dir / dsm_p.name, "w", **profile) as o:
            o.write(ndsm, 1)
        written += 1
    print(f"derive_ndsm: wrote {written} nDSM tiles to {out_dir}")
    return written


# --------------------------- prefetch (offline-first, like GAMUS) ---------------------------

def prefetch_open_canopy(dst, allow_patterns=None, retries: int = 4) -> None:
    """Download an Open-Canopy SLICE from HuggingFace (resumable via the HF cache).
    Pass allow_patterns to grab only a slice (the full set is ~360 GB), e.g.
    allow_patterns=["*/2021/*canopy*", "*/2021/*rgb*"] — refine to the real paths."""
    from huggingface_hub import snapshot_download
    import time
    for attempt in range(retries):
        try:
            snapshot_download(repo_id="AI4Forest/Open-Canopy", repo_type="dataset",
                              local_dir=str(dst), allow_patterns=allow_patterns)
            print("Open-Canopy slice cached.")
            return
        except Exception as e:  # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
            else:
                print(f"Open-Canopy prefetch FAILED: {type(e).__name__}: {e}")


def prefetch_m4heights(dst, allow_patterns=None, token=None, retries: int = 4) -> None:
    """Download an M4Heights slice from HuggingFace (GATED, resumable via HF cache).

    Requires (1) an HF token AND accepting the dataset licence on the repo page, and
    (2) allow_patterns to grab ONLY the 1 m aerial ortho + building-height archives
    (skip the Sentinel-1/2 bulk). The data is ZIPPED — after this, unzip the fetched
    archives into rgb/ + height/ (matched stems) before m4heights_source() reads them.
    Example: allow_patterns=["*AERIAL*", "*height*"] — refine to the real paths."""
    from huggingface_hub import snapshot_download
    import time
    for attempt in range(retries):
        try:
            snapshot_download(repo_id="Rituxx96x/M4Heights", repo_type="dataset",
                              local_dir=str(dst), allow_patterns=allow_patterns, token=token)
            print("M4Heights slice cached. NOW UNZIP the archives into rgb/ + height/.")
            return
        except Exception as e:  # noqa: BLE001
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
            else:
                print(f"M4Heights prefetch FAILED: {type(e).__name__}: {e} "
                      f"(gated — is your HF token set and the licence accepted?)")


def gbh_download_note() -> str:
    """GBH isn't a simple HF pull — it's on mediaTUM. Return the manual steps."""
    return ("GBH (OPTIONAL, license-gated): the code is at "
            "github.com/zhu-xlab/GlobalBuildingAtlas; the DATA (GBA.Height) is on "
            "mediaTUM (mediatum.ub.tum.de/1782307 = doi.org/10.14459/2025mp1782307), "
            "NOT HuggingFace. License is CC BY-NC (non-commercial) — CONFIRM the "
            "Height license before use. Then unpack RGB + nDSM GeoTIFFs into two "
            "folders and point gbh_source(root, rgb_subdir=..., height_subdir=...) at them.")
