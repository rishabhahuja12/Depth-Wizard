# Dataset Licenses & Attributions

Licensing status of every dataset the upgrade can train on. **Rule: keep the core
pipeline license-clean (commercial-OK) so the trained model carries no encumbrance;
CC BY-NC data is an OPTIONAL, informed opt-in only.**

| Dataset | License | Commercial use? | Role |
|---|---|---|---|
| **GAMUS** | CC BY 4.0 | ✅ yes (attribution) | **Core trainer** |
| **Open-Canopy** | open (see HF card) | ✅ yes | **Primary aux slice** |
| **GBH / GlobalBuildingAtlas (GBA.Height)** | **CC BY-NC 4.0** | ❌ **NON-commercial only** | **Optional, gated** |

## GBH — confirmed terms (from the GlobalBuildingAtlas web viewer ToU)
- **License:** Creative Commons Attribution-**NonCommercial** 4.0 (CC BY-NC 4.0) —
  https://creativecommons.org/licenses/by-nc/4.0/
- **"You may not use the material for commercial purposes."**
- Attribution required; citation required for academic/public use.
- Data host: mediaTUM (mediatum.ub.tum.de/1782307), **not HuggingFace**. The GitHub
  repo (zhu-xlab/GlobalBuildingAtlas) is code only.

### Decision (recorded)
- Core = **GAMUS + Open-Canopy** (both commercial-OK) → the model stays license-clean.
- **GBH is used ONLY IF** the entire DepthWizard/ISRO deliverable is confirmed
  non-commercial. Open question (do NOT assume resolved): whether a model *trained
  on* NC data inherits the NC restriction — CC's stance on ML training is contested,
  so the conservative reading is that it does. If in any doubt, **omit `--gbh-root`**
  and blend Open-Canopy alone.

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
