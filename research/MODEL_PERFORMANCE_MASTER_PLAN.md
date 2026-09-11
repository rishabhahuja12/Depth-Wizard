# DepthWizard — Model Performance Master Plan

> **Branch:** `research/depth-model-upgrade` · Derived from **`DEPTH_MODEL_RESEARCH_DUMP.md`**
> **Goal:** Maximize measured DSM/height accuracy and render quality, on a single **RTX 4500 Ada (24 GB)** workstation, without breaking the <1 s inference / 60 FPS render budgets.
> **Guiding principle:** *Measure first, then move the biggest lever.* Every phase has a numeric acceptance gate; nothing ships on vibes.

---

## 0. Objectives & success metrics

**North-star:** on a held-out GAMUS city (unseen geography), beat the current pipeline by a wide, honest margin.

| Metric (held-out city) | Current (est., to be measured in P0) | Target after P1 | Stretch after P2 |
|---|---|---|---|
| Height **MAE (m)** | unmeasured (baseline in P0) | **−30% or better** vs baseline | −40% |
| Height **RMSE (m)** | unmeasured | −25% | −35% |
| **δ1** (≤1.25) | unmeasured | +0.10 abs | +0.15 |
| **Boundary F-score** | unmeasured | +0.08 | +0.15 |
| Tall-building (>15 m) MAE | unmeasured | −35% (long-tail fix) | −45% |
| Inference — **Live Mode** (1024² tile) | ~250–650 ms | ≤ 900 ms single-pass | — |
| Inference — **Export tiling** (offline, gated) | n/a | ~4–5 s (9 ViT-L passes) | — |

> The "current unmeasured" cells are the whole reason P0 exists. The percentage targets are relative to whatever P0 records.

---

## 1. Phase map (dependencies & sequencing)

```
P0 Benchmark FIRST ─▶ P1 Metric fine-tune ───────────▶ P2 Detail/edges
  (measure current)    (un-normalize → DA2-L → meters)    Live ~350ms + Export tiling
        │                     │
        │                     ├─▶ P1b DEM base-terrain anchor (flood-critical)
        │                     ▼
        └──▶ P3a triplanar+sharpen (app-layer, parallel day 1) ─▶ P3b skirts
                              │
                              ▼
                        P4 Off-nadir (lighter-first)   P5 Super-res (gated, optional)
```

- **P0 blocks everything** (the measuring stick — and where the before/after table judges want comes from).
- **P1 Task 0 is un-normalization** — the pipeline currently deletes metric scale in 3 places; fix that (plus ViT-S→Large, `CombinedLoss` drift, 8 GB batch sizing) before training.
- **P1b (DEM anchor) is not optional** — without a real base terrain, mountains render flat and the flood sim breaks.
- **P3a (triplanar + sharpen)** is app-layer and can start **in parallel immediately** — no model dependency.
- **P2 tiling is Export-only** (~4–5 s); the interactive path stays a single-pass **Live Mode**.

---

## 2. Phase details

### P0 — Evaluation harness & baseline *(prerequisite, ~2–3 days)*
**Do:**
- Add `backend/evaluation/benchmark.py`: GAMUS held-out loader (hold out one full city, e.g. NYC), runner over the *existing* pipeline, and a metrics module (MAE/RMSE/δ1-3/SI-RMSE/boundary-F/long-tail bias) built on `validation.py`.
- Emit `eval_report.md` + `eval_metrics.csv` via one command `python -m backend.evaluation.benchmark`. **The before/after table it produces is the credibility slide for the jury** ("RMSE 3.8 m → 1.6 m on unseen tiles").
- **Verify GAMUS GSD** from the `EarthNets/RSI-MMSegmentation` dataloader and record it (feeds P1 scaling + P5 go/no-go).
- **Log the housekeeping blockers** the code audit found so P1 clears them first: metric normalization in 3 places (`dataset_gamus.py:187`, `train_gamus_full.py:299`, `depth_estimator.py:94`), `MODEL_ID` still ViT-S (`config.py:11`), `CombinedLoss(alpha=…)` signature drift (`train_gamus_full.py:267`), batch sized for the old 8 GB laptop.
**Deliverable:** baseline numbers for every §0 metric + a fix-list for P1.
**Gate:** harness reproduces a stable number twice; baseline table committed.

### P1 — Metric nDSM fine-tune of Depth Anything V2-Large *(the core win, ~1.5–2.5 weeks)*
**Task 0 — un-normalize the pipeline & clear housekeeping (prereq, ~1 day):** stop min-max/percentile normalizing the **target and prediction** (`dataset_gamus.py:187`, `train_gamus_full.py:299–303`, `losses.py:47`, `depth_estimator.py:94`) so real meters survive end-to-end; bump `MODEL_ID` → `Depth-Anything-V2-Large-hf`; reconcile the `CombinedLoss` constructor signature; re-tune batch/grad-accum for 24 GB. *Nothing else in P1 works until this lands — otherwise you just re-bake a relative model.*
**Do:**
- Data (dump §2.8): **GAMUS core** RGB→**nDSM-in-meters** pairs (no target normalization), 518² crops from 1024² tiles; *optional* blended slices for terrain diversity — **Open-Canopy** (forested) + **GBH** (global/sparse) into the same nDSM head. Radiometric stretch matched to `geospatial.py`; **appearance-only India-hardening augmentation** (shadow/sun-angle, haze, albedo, scale jitter — §2.7) + flips/rotations. **No tilt/perspective warps** (would corrupt nDSM).
- Model: DA2-**Large** (DINOv2-L + DPT); **retarget head to regress metric nDSM (meters)** + keep a relative path for non-georef PNGs.
- Losses, staged and each measured against P0: **(1)** SiLog + Smooth-L1 (metric) → **(2)** add **single-scale Sobel** edge term (lean-first; multi-scale/Laplacian only if boundary-F stalls — it's cheap on the output map, *not* an OOM risk) → **(3)** add HTC long-tail term. Optional virtual-normal.
- Recipe: differential LR (enc `5e-6`, head `5e-5`), AdamW, cosine schedule, AMP (bf16), gradient checkpointing, **batch 4 + grad-accum 4 (eff. 16)**, ~30–40 epochs. Early-stop on held-out MAE.
- Integrate winning checkpoint into `depth_estimator.py`; **repurpose `calibration.py`** — it no longer invents meters; it composes the AI's metric nDSM (see P1b) with a real DEM base + a thin relative-mode fallback.
- **Validate on real Indian tiles** (Cartosat/Sentinel/Bhuvan) qualitatively before claiming India-readiness (dump §2.7).
**VRAM plan (24 GB):** DA2-L full FT at 518² fits with checkpointing+bf16 (~14–18 GB); if tight, batch 2 + grad-accum 8, or LoRA (fits easily, frees VRAM). *(The edge loss is not the memory driver — the backbone is.)*
**Deliverable:** fine-tuned metric checkpoint + eval report vs baseline.
**Gate:** MAE −30% and boundary-F +0.08 on held-out city, no regression on relative-mode sanity tiles. *If Large plateaus badly → try Giant-LoRA as a bounded experiment.*

### P1b — Absolute DSM: DEM base-terrain anchor *(flood-critical, ~2–4 days)*
The GAMUS fine-tune outputs **nDSM (height above local ground)** — no sea-level datum, no natural slope. To make elevations absolute and the flood sim honest (dump §2A):
**Do:**
- For georeferenced tiles, fetch a coarse public DEM (**Copernicus GLO-30 / SRTM 30 m**; **CartoDEM** for India AOIs) over the tile's CRS/affine footprint, upsample to grid → `DTM_base`.
- Compose `DSM_absolute = DTM_base + ĥ_nDSM`; feed absolute elevations to the 3D viewer + flood simulator.
- Non-georeferenced PNGs: flat base / relative mode (no datum recoverable).
- Cache DEM tiles; **bundle an offline DEM for the demo AOIs** so the jury run never hits the network.
**Gate:** flood slider produces physically sensible inundation on a sloped-terrain tile (river fills the valley first, hills stay dry); mountains render with real relief, not flat.

### P2 — Detail & edge sharpness: Live Mode + Export tiling *(~1 week)*
**Do:**
- Keep a **Live Mode**: single 518² resized forward pass (~350 ms) — the default for upload/sliders so the demo stays snappy.
- Add **tile-based inference** (PatchFusion-/PRO-style overlapping 518² patches, consistency-blended) behind a **`[High-Precision GIS Tile Mode]` toggle** — ~9 ViT-L passes ≈ **3.5–5 s**, so it's **offline export**, not interaction.
- Confirm the P1 edge losses + tiling produce crisp discontinuities on the eval boundary-F metric.
**Deliverable:** two-tier inference (Live/Export) + updated eval.
**Gate:** Export tiling improves RMSE vs Live single-pass, seams invisible; Live Mode stays ≤ ~400 ms.

### P3 — Render-side mesh fix *(P3a can start day 1)*
**P3a (quick, ~1–2 days, parallel):** triplanar texturing + edge-aware depth sharpening in `TerrainCanvas.tsx`. Kills the smeared-curtain look.
**P3b (after P1, ~3–5 days):** insert **vertical skirt quads** at large depth discontinuities (now detectable thanks to sharp edges) → real walls.
**P3c (stretch):** LOD1 building extrusion via footprint segmentation — separate ground vs prismatic buildings.
**Gate (P3a):** side-by-side screenshots show no texture smearing on building faces; 60 FPS held. (P3b/P3c are visual-quality milestones, not blocking.)

### P4 — Off-nadir / tilted inputs *(scoped: detect + flag, NOT synthesize)*
**Two constraints kill the geometric and the data-side fixes:** no RPC → can't straighten; and **you cannot synthesize oblique (RGB, nDSM) pairs from nadir data** — a 2D warp breaks the height↔image correspondence (parallax is height-dependent) and a 3D re-render fabricates wall/occlusion pixels. **Oblique augmentation is therefore off the table — it would corrupt the model.**
**Do (honest, cheap):**
- Treat the model as **nadir-only**; add a lightweight **image-based tilt detector** → show "⚠ oblique — reduced accuracy" and degrade gracefully.
- Keep augmentation **appearance-only** (photometric / India-hardening) + flips/rotations/scale; **no tilt/perspective warps**.
**If real off-nadir support is ever needed (v2):** fine-tune on a **real** off-nadir dataset (**SpaceNet-4**), not synthesized data; MLS-BRN angle/offset heads optional there.
**Gate:** oblique inputs are correctly *flagged* and nadir accuracy is unaffected. (No claim of correcting oblique geometry.)

### P5 — Super-resolution ablation *(optional, gated, ~3–4 days)*
**Only if** P0 confirms a genuine low-res ingest need (e.g., Sentinel-2 10 m). Then: transformer SR front-end, measured on P0 harness **against a no-SR baseline**; ship only if it wins *without* inflating elevation error. Otherwise document as rejected. **Do not** super-resolve labels.

---

## 3. Compute & VRAM budget (RTX 4500 Ada, 24 GB)

| Workload | Fits? | Notes |
|---|---|---|
| DA2-**Large** full FT @518², bf16 + ckpt, bs 4–8 | ✅ | ~14–18 GB; primary path |
| DA2-Large LoRA @518² | ✅ easily | Fallback; frees VRAM for larger batch / faster A/B |
| DA2-**Giant** LoRA | ✅ (tight) | Only if Large plateaus; ~3× epoch time |
| Tiled inference (P2) | ✅ | N patches sequential; watch latency |
| Depth Pro full FT | ⚠️ | Possible but we only *borrow its losses*, not train it |

Estimated P1 train time: ~overnight-to-2-nights for 30–40 epochs on GAMUS at Large — well within a local workstation cadence (mirrors the repo's existing overnight-training workflow).

---

## 4. Workstream ordering (fastest path to visible value)
1. **Day 1:** kick off **P3a** (triplanar+sharpen — instant visual win) *and* start **P0** harness.
2. **Week 1:** P0 baseline locked; GSD verified; housekeeping fix-list from the audit.
3. **Weeks 2–4:** **P1** — Task 0 (un-normalize) first, then the metric fine-tune (the real prize), with India-hardening aug.
4. **Week 4–5:** **P1b** DEM anchor (unblocks honest flood sim) + **P3b** skirts on the new model.
5. **Week 5+:** **P2** Live/Export inference; validate on real Indian tiles.
6. **Later / as needed:** **P4** off-nadir (lighter-first), **P5** SR ablation, **P3c** LOD1.

## 5. Decision gates (kill/continue)
- **Before P1 training:** if Task 0 (un-normalization) isn't done, **do not train** — you'd only re-bake a relative model.
- After **P0**: if GAMUS GSD/labels are unusable → pause, source alternate nDSM data before P1.
- After **P1 loss-stage 1**: if metric SiLog+Smooth-L1 doesn't beat baseline MAE → debug data/scaling *before* adding edge/long-tail terms (don't stack losses on a broken base).
- After **P1**: if Large hits target → **do not** chase Giant (diminishing returns). If it misses → Giant-LoRA experiment.
- After **P1b**: if the flood sim still misbehaves on sloped terrain → the DEM anchor is wrong; fix before demo (it's the headline feature).
- Before **P5**: no low-res ingest need → **skip SR entirely** (research already argues against it).

## 6. What changes in the codebase (traceability)
- `backend/evaluation/benchmark.py` (new) — P0 harness + before/after tables.
- `backend/training/dataset_gamus.py` & `train_gamus_full.py` — **remove target/pred normalization** (regress meters, Task 0); **appearance-only** India-hardening augmentation (no tilt/perspective — that would corrupt nDSM labels); *optional* Open-Canopy / GBH loaders for terrain diversity (§2.8); fix the `CombinedLoss` call.
- `backend/training/losses.py` — metric Smooth-L1 + single-scale Sobel (add HTC long-tail later).
- `backend/app/config.py` — `MODEL_ID` → DA2-**Large**.
- `backend/app/services/depth_estimator.py` — load DA2-L metric checkpoint; **stop re-normalizing output**; Live Mode + Export tiling path (P2).
- `backend/app/services/calibration.py` — **repurpose**: compose real DEM `DTM_base` + metric nDSM (P1b); thin relative-mode fallback.
- `backend/app/services/geospatial.py` — DEM fetch/cache (P1b — needs only CRS/affine, **unaffected by the RPC gap**); optional image-based off-nadir tilt detection (P4); RPC orthorectification only *if* RPC-tagged tiles ever appear.
- `frontend/src/components/TerrainCanvas.tsx` — triplanar + sharpen (P3a), skirt geometry (P3b); Live/Export toggle wiring.

---

*Companion research and full citations: `research/DEPTH_MODEL_RESEARCH_DUMP.md`.*
