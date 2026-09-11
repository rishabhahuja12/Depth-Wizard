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
├── README.md            ← this file
├── .gitignore           ← ignores heavy local artifacts (checkpoints, data, reports)
├── evaluation/          ← P0: baseline harness + metrics + before/after tables
├── training/            ← P1: metric fine-tune scripts (Large, un-normalized, meters)
├── data_prep/           ← dataset harmonization (GAMUS + Open-Canopy + GBH) & DEM fetch (P1b)
├── outputs/             ← generated reports, metric CSVs, experiment logs   (git-ignored)
├── checkpoints/         ← trained model weights                             (git-ignored)
└── notes/               ← working notes, decisions, scratch
```

---

## Roadmap this folder implements

Source of truth: `research/MODEL_PERFORMANCE_MASTER_PLAN.md` (+ `DEPTH_MODEL_RESEARCH_DUMP.md`).

| Phase | What | Status |
|---|---|---|
| **P0** | Evaluation harness + baseline table (hold out one GAMUS city) | not started |
| **P1 Task 0** | Un-normalize pipeline · backbone → Large · fix drifted code | not started |
| **P1** | Retrain to metric meters — **GAMUS first**, then blend **Open-Canopy slice + GBH whole** (harmonized) | not started |
| **P1b** | Real DEM base terrain (Copernicus/SRTM + CartoDEM for India) | not started |
| **P2 / P3** | Detail/tiling · mesh "melted-tent" fix | not started |
| **P4** | Off-nadir = **detect + flag only** (stated limitation; no v2) | not started |
| **P5** | Super-resolution | likely skipped |

### Decisions locked in
- **Hardware:** RTX 4500 Ada, 24 GB → full Large fine-tune is on the table.
- **Goal:** genuine accuracy, no deadline → full plan, in order.
- **Datasets:** GAMUS core → then blend a slice of Open-Canopy + all of GBH (harmonized to a
  common ground resolution; GBH license still to verify).
- **Off-nadir:** detect + flag only. No v2 (instance building-reconstruction) — data-limited
  and out of scope for this effort.

### Verified facts
- **GAMUS GSD = 0.33 m/pixel** (arXiv:2305.14914, "resolution of 0.33m") — RGB + nDSM.
- **GAMUS labels = nDSM, meters above ground** (DSM − DTM) → no sea-level datum, which is
  exactly why P1b's DEM base anchor is required.
- **Held-out set = the `test` split**; the model was fine-tuned on `train`. Filenames encode
  city (`DC_…`, `PHL_…`) so per-city breakdown works.
