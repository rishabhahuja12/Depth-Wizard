# `upgrade/` — Depth-Model Upgrade Sandbox

All work from the **model-performance upgrade** (branch `research/depth-model-upgrade`) lives
inside this one folder. It is a **self-contained, disposable workspace**.

> **If we ever abandon this effort: delete this single folder.** Nothing outside it depends
> on anything inside it, so removing `upgrade/` leaves the original DepthWizard app fully
> intact and runnable.

---

## The one rule that makes this safe to delete

**Dependencies point one way only:**

```
upgrade/  ─────imports───▶  backend/ , frontend/      ✅ allowed
backend/  ──────X──────▶  upgrade/                    ❌ never
frontend/ ──────X──────▶  upgrade/                    ❌ never
```

Code in `upgrade/` may *read from and call into* the existing app (e.g. the evaluation
harness runs the current `depth_estimator`). The existing app must **never** import from
`upgrade/`. As long as that holds, `upgrade/` is always removable in one step.

### The one honest caveat
Some upgrade phases (un-normalizing the pipeline, swapping the backbone to Large, serving
the retrained model) must *eventually* edit files under `backend/` to take effect in the
live app. Until that final, deliberate integration step, **all development stays here** and
is fully disposable. Integration into the live app is a separate, explicitly-decided,
reversible action — not something that leaks in as a side effect of building.

---

## Planned layout (subfolders appear as each phase is built)

```
upgrade/
├── README.md            ← this file (overview + status)
├── SETUP.md             ← the workstation run guide
├── .gitignore           ← ignores heavy local artifacts (checkpoints, data, logs)
├── logging_config.py    ← shared structured/timestamped logging
├── docs/                ← reference docs: datasets guide, licences, integrations, workstation
├── evaluation/          ← P0: baseline harness + metrics + before/after tables
├── training/            ← P1: metric losses/dataset/loop, adapters, harmonize, prefetch
├── inference/           ← P1b DEM base · P2 tiling · P4 off-nadir detect
├── logs/                ← per-run training logs                             (git-ignored)
├── outputs/             ← generated reports, metric CSVs                    (git-ignored)
├── checkpoints/         ← trained model weights                             (git-ignored)
└── data/                ← prefetched GAMUS cache + manifests                (git-ignored)
```

---

## Roadmap this folder implements

Source of truth: `research/MODEL_PERFORMANCE_MASTER_PLAN.md` (+ `DEPTH_MODEL_RESEARCH_DUMP.md`).

| Phase | What | Status |
|---|---|---|
| **P0** | Evaluation harness + baseline table (held-out `test` split, per-city) | ✅ built & tested; full run pending on workstation |
| **P1 Task 0** | Audit of the metric-scale deletion points; fresh build in `upgrade/` (backend untouched) | ✅ audited (4 kill-points + broken `CombinedLoss` signature) |
| **P1 Stage 1** | Un-normalized metric losses + meters dataset + ViT-Large training loop | ✅ built & tested; smoke-verified; **train runs on workstation** |
| **P1 Stage 2/3** | Sobel edge + HTC long-tail losses (weight-gated) | ✅ **built & tested**; execution gated on Stage 1 beating baseline |
| **P1 adapters** | LoRA/DoRA (fallback / Giant-enabler) | ✅ **built & tested** |
| **P1 blend** | Open-Canopy source + `--blend` wired; shared reader hardened for more aux sources | ✅ **built, wired & CPU-integration-tested**; run after GAMUS-only works. Adopted flow: **GAMUS → +Open-Canopy (forest) → +M4Heights (urban, easy) → +DFC2023 (global diversity) → NL pair-builder (quality ceiling, if needed)**. GBH dropped (paired RGB not public); US3D/GeoNRW on hold/fallback. Full rationale in `docs/DATASETS_GUIDE.md`; licences in `docs/DATASET_LICENSES.md` |
| **P1b** | Absolute DSM = real DEM base + nDSM (flood-critical) | ✅ **built & tested** (compose + DEM upsample; offline DEM adapter) |
| **P2** | Tile-based hi-res inference (Export path; Live stays single-pass) | ✅ **built & tested** (seamless stitch) |
| **P4** | Off-nadir detect + flag (anisotropy cue; never correct/synthesize) | ✅ **built & tested** |
| **P1b** | Real DEM base terrain (Copernicus/SRTM + CartoDEM for India) | not started |
| **P2** | Detail/tiling (Live vs Export) | not started |
| **P3a** | Triplanar texturing | ❌ tried & reverted — wrong lever (see docs/INTEGRATIONS.md) |
| **P3 (mesh)** | Melted-tent fix | **deferred post-P1**; voxel mode is the primary renderer; skirts (P3b) revisited after P1 sharpens edges |
| **P4** | Off-nadir = **detect + flag only** (stated limitation; no v2) | not started |
| **P5** | Super-resolution | likely skipped |

### Decisions locked in
- **Hardware:** RTX 4500 Ada, 24 GB → full Large fine-tune is on the table.
- **Goal:** genuine accuracy, no deadline → full plan, in order.
- **Datasets:** GAMUS core → gated blend **+Open-Canopy (forest) → +M4Heights (urban) →
  +DFC2023 (global diversity) → NL pair-builder (0.5 m LiDAR quality, if needed)**, each
  harmonized to a common ground resolution and each gated on beating val MAE. GBH dropped
  (paired RGB not public); US3D/DFC2019 + GeoNRW on hold as fallbacks. Plain-English
  rationale in `docs/DATASETS_GUIDE.md`.
- **Off-nadir:** detect + flag only. No v2 (instance building-reconstruction) — data-limited
  and out of scope for this effort.

### Verified facts
- **GAMUS GSD = 0.33 m/pixel** (arXiv:2305.14914, "resolution of 0.33m") — RGB + nDSM.
- **GAMUS labels = nDSM, meters above ground** (DSM − DTM) → no sea-level datum, which is
  exactly why P1b's DEM base anchor is required.
- **Held-out set = the `test` split**; the model was fine-tuned on `train`. Filenames encode
  city (`DC_…`, `PHL_…`) so per-city breakdown works.
