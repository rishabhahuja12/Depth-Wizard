# Dataset Licenses & Attributions

Licensing status of every dataset the upgrade can train on. **Rule: keep the core
pipeline license-clean (commercial-OK) so the trained model carries no encumbrance;
CC BY-NC data is an OPTIONAL, informed opt-in only.**

| Dataset | License | Commercial use? | Role |
|---|---|---|---|
| **GAMUS** | CC BY 4.0 | ✅ yes (attribution) | **Core trainer** |
| **Open-Canopy** | open (see HF card) | ✅ yes | **Primary aux slice** |
| **GBH / GlobalBuildingAtlas (GBA.Height)** | **CC BY-NC 4.0** | ❌ non-commercial only | License cleared ✅ · **usability TBD** (paired RGB?) |

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
- Core remains **GAMUS + Open-Canopy** (commercial-OK); GBH is an accepted addition
  under the non-commercial commitment. `--gbh-root` opt-in still applies.

### ⚠️ Open USABILITY question for GBH (license ≠ usable)
The license is cleared, but before relying on GBH verify it can actually feed our
**RGB → height** task:
- **GBA.Height on mediaTUM is the height product (nDSM GeoTIFF tiles, indexed by
  `height_tif.geojson` / `height_zip.geojson`).** Confirm it ships **co-registered
  optical RGB** paired with each height tile. If it is height rasters ONLY, it is
  NOT a drop-in training source for us (we need RGB↔height pairs).
- The HTC-DC Net training imagery is **PLANET (PlanetScope)** — **proprietary**.
  Planet's own license likely prevents open redistribution of the raw RGB, which is
  plausibly why only the heights are public. If the paired RGB isn't in the release,
  sourcing PLANET imagery separately carries its **own commercial license** — a
  second blocker independent of CC BY-NC.
- **Action:** inspect a mediaTUM GBA.Height tile set for paired RGB before wiring a
  real GBH blend. If absent → **drop GBH, blend Open-Canopy alone** (still the safe
  default). Open-Canopy is confirmed to ship paired SPOT RGB + canopy height.

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
