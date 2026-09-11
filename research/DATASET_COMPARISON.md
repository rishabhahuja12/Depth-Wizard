# DepthWizard — Dataset Comparison Sheet

> Reference for every dataset considered for the monocular height / DSM pipeline (SIH26175).
> Companion to `DEPTH_MODEL_RESEARCH_DUMP.md` §2.8. A spreadsheet version lives beside this file: **`dataset_comparison.csv`**.
>
> **Legend:** ✅ selected · ⏸️ optional / v2 · ❌ rejected · **⚠ verify** = value not authoritatively confirmed, check the source before relying on it.
> **Key idea:** every training label below is *height above ground* (nDSM / canopy height), so they share one regression head. Absolute elevation comes from the **DEM bases** (second table), not the model.

---

## A. Image → height training / label datasets

| Dataset | Provider · Platform | Imagery (sensor) | GSD | Tile size | Samples | Label type | Label source | Geography | Landscapes | License | Format · Size | Role | Key caveat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **GAMUS** | Earthflow · Hugging Face | Aerial ortho RGB | **⚠ ~0.3–0.5 m** | 1024×1024 | ~11,507 tiles | nDSM (height AGL), meters | Airborne LiDAR | 5 US cities (Oklahoma, DC, Philadelphia, Jacksonville, NYC) | Urban / suburban | CC-BY-4.0 | HDF5 · ~80 GB | ✅ **Core trainer** (recommended set) | US-only morphology; GSD unconfirmed |
| **DFC2019 / US3D** | IEEE GRSS · JHU-APL | WorldView-3 satellite (multi-date) | ~1.3 m | 1024×1024 (often cropped to 256) | ~2,783 triplets (~31k/4.4k/8.9k patches) | nDSM + DSM + semantic | Airborne LiDAR | Jacksonville FL, Omaha NE (US) | Urban / suburban | Open (contest) | GeoTIFF | ⏸️ Optional — matches **satellite-input** domain | US; overlaps GAMUS |
| **Open-Canopy** | IGN / LiDAR-HD · France | SPOT 6/7 satellite RGB | 1.5 m | ⚠ verify | 87,000 km² coverage | Canopy height (= vegetation nDSM) | Aerial LiDAR-HD | France | **Forested** | Open | Raster | ⏸️ Optional — **forested** slice | French forests; 1.5 m GSD |
| **GBH (Global Building Height)** | Academic (HTC-DC refs) | PLANET optical | ~3 m | ⚠ verify | 19 cities (2013–2021) | nDSM + building footprints | Open LiDAR | 19 cities worldwide | Urban / **sparse / global** | ⚠ verify | Raster | ⏸️ Optional — **global/sparse** diversity, narrows India gap | ~3 m GSD (coarse) |
| **ISPRS Vaihingen / Potsdam** | ISPRS | Aerial IRRG / RGB | 0.09 m / 0.05 m | ~2000² tiles | 33 (Vaih.) | DSM / nDSM | Dense image matching | Germany (urban) | Urban | Open benchmark | TIFF | ⏸️ Optional — ultra-hi-res **edge-sharpness** reference | Tiny; European urban only |
| **ETH Global Canopy Height** | ETH Zürich | Sentinel-2 + GEDI | 10 m | Global raster | Global | Canopy height | GEDI spaceborne LiDAR | Global | Forested | Open | GeoTIFF | ⏸️ Alt canopy (coarse) | 10 m too coarse for buildings |
| **SpaceNet-4 (off-nadir)** | SpaceNet · CosmiQ | WorldView-2, 7°–54° off-nadir | ~0.5 m | ~900² | ~126k footprints | Building **footprints** (no dense nDSM) | Manual + LiDAR | Atlanta (US) | Urban | CC-BY-SA (⚠ verify) | GeoTIFF | ⏸️ **v2 only** — the honest way to get real off-nadir signal | Footprints, not dense height; US |
| **Synthetic oblique from GAMUS** | (derived) | Warp / re-render GAMUS | — | — | — | nDSM (fabricated) | — | — | — | — | — | ❌ **Rejected** | Corrupts labels (parallax is height-dependent) / fabricates walls & occlusions |

---

## B. DEM / elevation bases (for **absolute** anchoring — not model training)

| DEM | Provider | Type | Posting / GSD | Coverage | Vertical accuracy | Role | Note |
|---|---|---|---|---|---|---|---|
| **Copernicus GLO-30** | ESA / Airbus (TanDEM-X) | DSM | 30 m | Global | ~2–4 m (typ.) | ✅ **Preferred global base** | Includes surface (DSM) |
| **SRTM 30 m** | NASA / USGS | DSM (radar) | 30 m (1 arc-sec) | Near-global (±60° lat) | ~16 m LE90 (often better) | ✅ Global base (**brief-blessed**) | Voids in steep terrain |
| **CartoDEM v3** | ISRO / NRSC · Bhuvan | DEM (optical stereo) | 30 m (1 arc-sec) | **India** | Terrain RMSE ~1.2–1.8 m (vs ICESat-2) | ✅ **India base + validation reference** | ISRO's own product — good jury optics; India only |

---

## How to read this
- **Training strategy (§2.8):** GAMUS is the core; add **Open-Canopy** (forested) and optionally **GBH** (global/sparse) *only if time allows* — same nDSM head. Do **not** replace GAMUS.
- **Absolute elevation:** Copernicus GLO-30 / SRTM globally; **CartoDEM for India**.
- **Off-nadir:** synthetic oblique is rejected (corrupts labels); real off-nadir (SpaceNet-4) is v2 only.
- **Before P1:** pin the two `⚠ verify` values that actually affect scaling — **GAMUS GSD** (from the `EarthNets/RSI-MMSegmentation` dataloader) and **GBH GSD/license** (only if you decide to blend it).

*Sources: see `DEPTH_MODEL_RESEARCH_DUMP.md` §9 References.*
