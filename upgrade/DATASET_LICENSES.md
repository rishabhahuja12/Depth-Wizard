# Dataset Licenses & Attributions

Licensing status of every dataset the upgrade can train on. **Rule: keep the core
pipeline license-clean (commercial-OK) so the trained model carries no encumbrance;
CC BY-NC data is an OPTIONAL, informed opt-in only.**

> **Plain-English rationale + how each dataset contributes:** see `DATASETS_GUIDE.md`.
> **Adopted training flow (gated on val MAE):** GAMUS (core) → +Open-Canopy (forest)
> → +M4Heights (urban, HF-easy) → +DFC2023 (global diversity) → NL pair-builder
> (0.5 m LiDAR quality ceiling, only if needed). US3D/DFC2019 + GeoNRW are on hold as
> fallbacks. Principle: each aux source must own a distinct eval landscape
> (urban/sparse/hilly/forested) — breadth buys diversity, not more urban.

| Dataset | License | Commercial use? | Role |
|---|---|---|---|
| **GAMUS** | CC BY 4.0 | ✅ yes (attribution) | **Core trainer** |
| **Open-Canopy** | open (see HF card) | ✅ yes | **Primary aux — forest/canopy height** |
| **US3D / DFC2019** | open-access (IEEE DataPort) | ⚠ confirm | Aux — buildings (WorldView-3 satellite, US). Adapter **ON HOLD** |
| **M4Heights** | CC-BY (confirm on HF repo) | ✅ likely | Aux — buildings (1 m aerial, NL/DE/CH). HF-pullable. Adapter **ON HOLD** |
| **DFC2023 Track 2** | research-use (contest registration) | ⚠ no/unclear | Aux — buildings (global, 12 cities / 5 continents). Adapter **ON HOLD** |
| **GeoNRW** | dl-de/by-2.0 (attribution) | ✅ yes | Aux — buildings (1 m aerial, NRW). Code written, **ON HOLD** (ships DSM, needs DTM→nDSM) |
| **GBH / GlobalBuildingAtlas (GBA.Height)** | CC BY-NC 4.0 | ❌ non-commercial only | ❌ **DROPPED — paired RGB not publicly available (PLANET proprietary)** |
| **Huawei BHE** | unverified | ⚠ unknown | ❌ **DROPPED — availability + license unconfirmed** |

> **Adapter status:** the shared reader (`GeoTiffHeightSource`) already handles every
> per-dataset quirk via parameters — nodata/scale (`sanitize_height`), `_RGB`↔`_AGL`
> name remap (`rgb_token`/`height_token`), multi-band imagery (`rgb_bands`), and
> DSM→nDSM (`derive_ndsm`). The **per-dataset `*_source()` adapters + `--root` flags
> for M4Heights / DFC2023 / US3D are now WRITTEN and wired** (`--m4h-root` / `--dfc-root`
> / `--us3d-root`); what remains per source is the actual **data download + confirming
> the real folder layout** (and for M4Heights, an unzip; for US3D, a stem rule if a tile
> has many views). GeoNRW's adapter + `derive_ndsm` are written but dormant until a DTM.

### Acquisition (how each is pulled)
| Dataset | HF mirror | Channel |
|---|---|---|
| M4Heights | ✅ `Rituxx96x/M4Heights` | HF `snapshot_download` (grab 1 m aerial + height slice) |
| GeoNRW | ✅ `torchgeo/geonrw` (32 GB) | HF pull — **but no DTM**, so nDSM needs a separate NRW DGM |
| US3D / DFC2019 | ❌ none | Manual: IEEE DataPort (free login) or `pubgeo/datasets` |
| DFC2023 T2 | ❌ none | Manual: IEEE DataPort (contest registration) |

### National LiDAR + ortho programs (candidate pair sources — under evaluation, NOT decided)
Country open-data programs publish **0.5–1 m DSM + DTM rasters** and, separately,
**open aerial orthophotos** — so full RGB→nDSM pairs are *constructible* (open ortho +
`derive_ndsm(DSM, DTM)`), same pattern as GeoNRW. The vector **LoD2 3D building
models** (3DBAG, LoD2-DE, swissBUILDINGS3D) are a *different* task from our dense
raster regression — not used here.
- **Netherlands** — AHN (0.5 m DSM+DTM, open) + PDOK aerial ortho + 3DBAG. Highest quality.
- **USA** — NAIP ortho (public domain) + USGS 3DEP (1 m DSM/DTM); OpenTopography hosts some pre-made nDSM. Aerial (closest to GAMUS domain).
- **Germany** — LoD2-DE / state DOP + DTM/DSM (1 m). GeoNRW = the NRW instance.
- **Switzerland** — swissALTI3D (DTM) + swissSURFACE3D (DSM) + swissimage ortho (0.5 m).

### India (ISRO deliverable context — mostly targets/eval, not RGB pairs)
- **Google Open Buildings 2.5D / AEEE GOBS** — footprint + AGL height, **no paired optical** (imagery proprietary). Use as an **eval/fine-tune target**, not a training pair — same trap as GBH.
- **GlobalBuildingAtlas (India)** — same proprietary-imagery limitation as GBH.
- **ISRO Bhuvan CartoDEM (30 m)** — too coarse for buildings; useful as the **DEM base for inference** (`inference/dem_base.py`: absolute DSM = DEM base + nDSM), not for training.
- High-res Indian DSM/DTM is gated (Survey of India) → strategy is train on open Western pairs, adapt/evaluate on India.

## GBH — confirmed terms (from the GlobalBuildingAtlas web viewer ToU)
- **License:** Creative Commons Attribution-**NonCommercial** 4.0 (CC BY-NC 4.0) —
  https://creativecommons.org/licenses/by-nc/4.0/
- **"You may not use the material for commercial purposes."**
- Attribution required; citation required for academic/public use.
- Data host: mediaTUM (mediatum.ub.tum.de/1782307), **not HuggingFace**. The GitHub
  repo (zhu-xlab/GlobalBuildingAtlas) is code only.

### Decision (recorded)
- **The DepthWizard / ISRO SAC deliverable is confirmed NON-COMMERCIAL** (project
  owner, this branch). → **GBH is CLEARED for use** alongside GAMUS + Open-Canopy.
- **Obligations that come with using GBH (must honor):**
  1. **Keep the deliverable non-commercial.** If it ever moves toward commercial
     use, GBH (and any model trained on it) must be removed / retrained without it —
     a model trained on NC data conservatively inherits the NC restriction.
  2. **Attribution + citation:** include the GBA citation (below) in any report,
     slide, or public presentation that uses a GBH-trained model.
- Core remains **GAMUS + Open-Canopy** (commercial-OK). **GBH is dropped** — not for
  licensing (that cleared) but because its paired RGB↔height training data is not
  publicly available (PLANET proprietary). See the verified usability note below.

### GBH USABILITY — VERIFIED, and it's DROPPED (data not available)
License was cleared (non-commercial), but a usability check settled it against GBH:
- HuggingFace `GBA.LoD1` / `GBA.ODbLPolygon` = **polygons + LoD1 JSON only, no imagery**.
- `GBA.Height` / `GBH` are **not on HuggingFace** (404); GBA.Height lives on mediaTUM
  and is the global height **product** (nDSM rasters) — **no paired optical RGB**.
- The HTC-DC-Net repo (the `im2bh` submodule) provides code + "organize your own
  data" — **no dataset download**.
- GBH imagery is **PLANET PlanetScope (proprietary)**; it cannot be openly
  redistributed, which is why the paired (RGB + nDSM) training set isn't public.
- **Conclusion:** the paired RGB↔height data we'd need is **not publicly available**,
  so GBH **cannot** feed our RGB→height task. Even self-sourcing PLANET carries its
  own **commercial** license — a hard conflict with the non-commercial deliverable.
  → **GBH dropped. Blend Open-Canopy alone** (ships paired SPOT RGB + canopy height,
  HF, open license). The `gbh_source()` adapter stays in code for anyone who has
  licensed paired rasters, but is not part of the plan.

### Code license note
The GlobalBuildingAtlas **code** is MIT + Commons Clause (no commercial use). We did
**not** use their code — our HTC-style long-tail loss is our own implementation of
the *idea* from the paper, so there is no code-license entanglement.

### Required GBH citation (if used)
```bibtex
@Article{globalbuildingatlas,
  AUTHOR  = {Zhu, X. X. and Chen, S. and Zhang, F. and Shi, Y. and Wang, Y.},
  TITLE   = {GlobalBuildingAtlas: an open global and complete dataset of building
             polygons, heights and LoD1 3D models},
  JOURNAL = {Earth System Science Data},
  VOLUME  = {17}, YEAR = {2025}, NUMBER = {12}, PAGES = {6647--6668},
  URL     = {https://essd.copernicus.org/articles/17/6647/2025/},
  DOI     = {10.5194/essd-17-6647-2025}
}
```
