"""
Auxiliary height datasets — Open-Canopy (forested) and GBH (global/sparse).

Both are paired RGB + height-in-meters GeoTIFFs, so both plug into the tested
GeoTiffHeightSource and mix through MixedMetricDataset onto the SAME height head
(research §2.8/§2.9). Sequencing: blend these ONLY after GAMUS-alone beats the
baseline.

Confirmed specs (see SETUP / research DATASET_COMPARISON):
  * Open-Canopy — HF AI4Forest/Open-Canopy, GeoTIFF, 1.5 m, France; SPOT RGB +
    LiDAR canopy-height (meters). ~360 GB — slice only.
  * GBH — mediaTUM (zhu-xlab/GlobalBuildingAtlas), 256x256 patches, 3 m; PLANET
    RGB + nDSM (meters). Data license: CONFIRM on mediaTUM before use.

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


def open_canopy_source(root, rgb_subdir: str = "images", height_subdir: str = "canopy_height",
                       rgb_glob: str = "*.tif") -> GeoTiffHeightSource:
    """Open-Canopy as a height source. Adjust the subdir names to the real download
    layout (canopy height is already meters-above-ground = vegetation nDSM)."""
    root = Path(root)
    return GeoTiffHeightSource(name="open_canopy", gsd=OPEN_CANOPY_GSD,
                               rgb_dir=root / rgb_subdir, height_dir=root / height_subdir,
                               rgb_glob=rgb_glob)


def gbh_source(root, rgb_subdir: str = "rgb", height_subdir: str = "ndsm",
               rgb_glob: str = "*.tif") -> GeoTiffHeightSource:
    """GBH as a height source (nDSM is meters-above-ground). Adjust subdirs to the
    real mediaTUM layout; confirm the data license before use."""
    root = Path(root)
    return GeoTiffHeightSource(name="gbh", gsd=GBH_GSD,
                               rgb_dir=root / rgb_subdir, height_dir=root / height_subdir,
                               rgb_glob=rgb_glob)


def build_aux_sources(oc_root=None, gbh_root=None, **kw) -> list:
    """Assemble whichever auxiliary sources have a root provided and actually
    contain paired tiles. Skips silently if a root is missing/empty."""
    sources = []
    if oc_root:
        s = open_canopy_source(oc_root, **{k[3:]: v for k, v in kw.items() if k.startswith("oc_")})
        if len(s) > 0:
            sources.append(s)
    if gbh_root:
        s = gbh_source(gbh_root, **{k[4:]: v for k, v in kw.items() if k.startswith("gbh_")})
        if len(s) > 0:
            sources.append(s)
    return sources


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


def gbh_download_note() -> str:
    """GBH isn't a simple HF pull — it's on mediaTUM. Return the manual steps."""
    return ("GBH: download patches from mediaTUM (doi.org/10.14459/2025mp1782307), "
            "confirm the data license, unpack RGB + nDSM GeoTIFFs into two folders, "
            "then point gbh_source(root, rgb_subdir=..., height_subdir=...) at them.")
