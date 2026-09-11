# Dataset Licenses & Attributions

Licensing status of every dataset the upgrade can train on. **Rule: keep the core
pipeline license-clean (commercial-OK) so the trained model carries no encumbrance;
CC BY-NC data is an OPTIONAL, informed opt-in only.**

| Dataset | License | Commercial use? | Role |
|---|---|---|---|
| **GAMUS** | CC BY 4.0 | ✅ yes (attribution) | **Core trainer** |
| **Open-Canopy** | open (see HF card) | ✅ yes | **Primary aux slice** |
| **GBH / GlobalBuildingAtlas (GBA.Height)** | CC BY-NC 4.0 | ❌ non-commercial only | ❌ **DROPPED — paired RGB not publicly available (PLANET proprietary)** |

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
