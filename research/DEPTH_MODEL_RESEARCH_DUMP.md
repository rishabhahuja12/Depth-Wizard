# DepthWizard — Deep Research Dump: Maximizing Monocular DSM Model Performance

> **Branch:** `research/depth-model-upgrade`
> **Author:** Research pass (Claude Opus 4.8) · September 2026
> **Purpose:** Evidence-backed survey of how to raise DepthWizard's core model accuracy and the surrounding application quality. Every recommendation states *why it was chosen*, the *trade-off vs the alternative we did not pick*, and the *evidence* behind it. A companion **`MODEL_PERFORMANCE_MASTER_PLAN.md`** turns this into a phased build plan.
>
> **Hardware envelope (confirmed):** RTX 4500 Ada (24 GB VRAM), Core Ultra 9 285, 36 GB RAM. Local workstation, no mandatory cloud. This is enough for **ViT-Large full fine-tunes** with gradient checkpointing/AMP, and **ViT-Giant via LoRA**.

---

## 0. TL;DR — the five findings that matter

1. **The single biggest win is not a bigger model — it is changing *what we predict*.** Today we predict *relative* disparity `[0,1]` and then rebuild metric height with hand-crafted morphology in `calibration.py`. That hand-crafted step is the dominant, uncontrolled error source. We have **real LiDAR nDSM labels in GAMUS**, so we should **fine-tune the network to regress metric height (nDSM in meters) directly** and **repurpose** `calibration.py` (hold a *real* DEM base + add the AI's metric nDSM — §2A) rather than delete it. Precedent: **Sat3R** did exactly this fine-tune (metric SiLog on Depth Anything V2) and cut MAE **~38%** vs zero-shot, approaching hours-long optimization methods at **~300× the speed** — and they only had *pseudo*-depth, not true LiDAR. [Sat3R] **Code-audit correction:** the pipeline doesn't merely *fail* to use meters — it **deletes** them in three stages (target normalized to `[0,1]` in `dataset_gamus.py:187`; pred+target min-max normalized per patch in `train_gamus_full.py:299–303`; inference re-normalized in `depth_estimator.py:94`; loss clamped to `[0,1]` in `losses.py:47`). So today's `best_model.pth` is a **relative-depth adaptation, not metric**, and the literal first task is to *un-normalize the pipeline end-to-end* (see the audit, §6A) before touching a bigger model.
2. **Right-size the backbone to ViT-Large, not Giant.** Depth Anything V2 **Large (DINOv2-L + DPT)** is the accuracy/effort sweet spot on 24 GB. Giant (1.3 B) gains little for aerial once fine-tuned and roughly triples train cost. **Depth Pro** is the model to steal *ideas* from (multi-scale patches + boundary losses) rather than adopt wholesale.
3. **Off-nadir/tilted imagery is a geometry problem, not a depth problem.** For images with RPC/RPB metadata, **rectify first**. When metadata is absent, the SOTA is to **predict the off-nadir & roof-offset angles as auxiliary heads** (MLS-BRN) and to **augment training with synthetic oblique views**. Do not expect a nadir-trained regressor to silently absorb 30°+ obliquity.
4. **Super-resolving the training data is the *weakest* of the four levers — do not lead with it.** Evidence is mixed (≈4–6% downstream gains at best, and *frequent regressions*), and SR **hallucinates high-frequency texture that a metric model will convert into fake elevation**. The better route to "more detail" is **native-resolution tile-based inference** (PatchFusion / "One Look is Enough"/PRO), which gives large RMSE gains *without* inventing pixels. Keep SR as a **gated ablation**, not a foundation.
5. **The "melted-tent" mesh is an application-layer artifact with a known fix ladder.** Three compounding causes (continuous mesh can't make vertical walls; monocular depth edges are blurry; a single draped texture stretches over the diagonal faces). Fixes, cheapest→deepest: **triplanar texturing → edge-aware depth sharpening → vertical "skirt" geometry at depth discontinuities → LOD1 building extrusion**. The Voxel mode already dodges this because quantized blocks *have* vertical faces.

### 0.1 Seven corrections after grounding in the repo + a lean-delivery review
The first draft was academically right but needed these adjustments once checked against the actual code and a hackathon-delivery lens. Each is expanded in the linked section.

1. **The metric signal is deleted *on purpose*, not just unused** — un-normalizing the pipeline is literally task #1, ahead of any bigger model. → §6A audit.
2. **Don't delete `calibration.py` — repurpose it.** GAMUS labels are *nDSM* (height **above local ground**), which carry no sea-level datum and no natural terrain slope. Delete the base-terrain model and mountains go flat and the **flood simulator breaks**. Keep a real `DTM_base` from a public DEM. → §2A.
3. **Keep the edge loss lean: single-scale Sobel + Smooth-L1 first.** (Note: the multi-scale/Laplacian variant is *not* a real VRAM risk — it runs on the single-channel output map, not the ViT-L features. The real VRAM lever is the backbone. But lean-first is still correct staging.) → §2.5.
4. **US→India domain gap is a first-class risk.** GAMUS = 5 US cities (tidy grids); ISRO = dense, irregular Indian terrain. Heavy radiometric/spatial augmentation + validate on real Cartosat/Sentinel tiles. → §2.7.
5. **Tiled hi-res inference is an *offline/export* feature, not the live path.** 9 ViT-L passes ≈ 3.5–5 s, not 1.5 s. Ship a ~350 ms single-pass **Live Mode**; gate tiling behind a **High-Precision/Export** toggle. → §4.2.
6. **Off-nadir: scope + detect + flag, never synthesize.** No RPC (can't straighten) *and* you can't fake oblique (RGB, nDSM) pairs from nadir data (a warp corrupts the label; a re-render fabricates walls/occlusions). So: nadir-only model + an "⚠ oblique" badge + appearance-only augmentation. Real off-nadir support (SpaceNet-4 / MLS-BRN heads) is a v2 that needs real data. → §3.3.
7. **Almost none of this is built yet** — see the implementation audit (§6A) and the loose-ends the audit found (§8.1).

---

## 1. What "accuracy" means here (so we can maximize it)

We cannot improve what we don't measure, and the repo currently has **no held-out quantitative depth benchmark wired into CI** (only `validation.py` with RMSE/δ helpers and qualitative browser screenshots). Before any model work, we standardize on the remote-sensing height metrics used across the literature (HTC-DC Net, THE-Benchmark, GAMUS):

| Metric | Definition | Why it matters for us |
|---|---|---|
| **MAE / RMSE (m)** | Mean abs / root-mean-square height error vs nDSM | Headline "how many meters off" number; RMSE punishes tall-building blunders |
| **δ1 / δ2 / δ3** | % pixels with `max(ŷ/y, y/ŷ) < 1.25, 1.25², 1.25³` | Standard depth accuracy; comparable to depth papers |
| **SI-RMSE / SiLog** | Scale-invariant log error | Fair when scale is ambiguous (relative mode) |
| **Boundary F-score / edge accuracy** | Precision-recall on depth discontinuities | Directly captures the "crisp building walls" quality the mesh needs |
| **Zero-height / background bias** | Error on ground pixels (the long-tail "head") | HTC-DC Net shows naive regressors systematically *underestimate* tall structures because ground dominates the loss |
| **Completeness** | % of DSM validly reconstructed | Matters once we back-project to a surface |

**Current-pipeline weaknesses this exposes:**
- Relative `[0,1]` output + morphological `DTM_base + α·nDSM` → the metric numbers shown in the UI are only *plausible*, not *validated* against LiDAR. `α` is derived from percentile spans, not learned.
- ViT-**S** (24.8 M) is the smallest backbone; it blurs building edges (root cause #2 of the mesh artifact).
- No boundary-aware loss → soft edges → tent/curtain artifacts on the smooth mesh.
- No off-nadir handling at all (camera assumed nadir).

---

## 2. Thread A — Backbone & fine-tuning (the core lever)

### 2.1 Current state and its ceiling
`depth_estimator.py` runs `Depth-Anything-V2-Small-hf` producing normalized relative disparity; `calibration.py` then *invents* metric height via a morphological min-filter DTM and a percentile-based `α`. This is a reasonable MVP but has three ceilings: **(a)** the network never sees a meter, so it can't learn metric priors (building storeys, road flatness); **(b)** ViT-S under-resolves edges; **(c)** the calibration is deterministic hand-tuning that can't adapt per-scene.

### 2.2 Candidate models considered

| Model | Params / backbone | Native output | Why interesting | Why *not* the base (trade-off) |
|---|---|---|---|---|
| **Depth Anything V2 – Large** ✅ *selected base* | 335 M · DINOv2-L + DPT | Relative; metric heads exist | Best-supported fine-tune path, strong DINOv2 features, HF + official metric recipe, fits 24 GB full FT | Edges softer than Depth Pro (mitigated by loss terms) |
| Depth Anything V2 – Giant | 1.3 B · DINOv2-g | Relative | Highest raw capacity | ~3× train cost, LoRA-only on 24 GB, marginal aerial gain once fine-tuned |
| **Depth Pro** (Apple) | 504 M · multi-scale ViT | **Metric, sharp, 1536²** | **Sharpest boundaries** (multi-scale patch enc + gradient/Laplacian losses + edge NMS); metric w/o intrinsics | Harder to fine-tune, less community tooling; **borrow its ideas** into DA2 |
| Metric3D v2 | ViT-L/g + canonical-camera | **Metric + normals**, zero-shot #1 | Canonical camera space solves metric ambiguity across sensors; surface normals for free | Camera-canonicalization assumes a pinhole intrinsic; overhead ortho/RPC doesn't fit cleanly |
| UniDepth v2 | ViT-L | Metric, self-promptable camera | Predicts its own camera; robust zero-shot | Same pinhole-centric assumption; extra complexity we don't need with LiDAR GT |
| Marigold | SD-based diffusion | Affine-invariant | Beautiful detail | Slow multi-step inference; affine-invariant (not metric); wrong tool for a <1 s app |
| HTC-DC Net | ViT enc + HTC-AdaBins | **nDSM height, classification-regression** | *Purpose-built for our exact task*; fixes long-tail height underestimation via head-tail-cut + distribution constraints | Smaller community; **adopt its loss/head design** on top of DA2-L rather than its codebase |

### 2.3 **Decision: fine-tune Depth Anything V2-Large to regress metric nDSM directly**, with an HTC-style long-tail-aware head.

**Why this pick over the alternatives:**
- **vs staying on ViT-S:** Sat3R's 38% MAE cut came from *metric fine-tuning*, and larger DINOv2 features measurably sharpen structure — both directly attack our two biggest errors. Cost is a one-time local train we can afford on 24 GB.
- **vs Giant:** Aerial nDSM is lower-entropy than open-world depth; the extra 1 B params mostly buy generalization we get from fine-tuning on in-domain LiDAR. Giant forces LoRA-only and ~3× epochs for a fraction of a meter. Deferred, not rejected — revisit if Large plateaus.
- **vs Metric3D v2 / UniDepth:** Their headline trick (canonical camera / self-predicted intrinsics) is designed for *unknown pinhole cameras*. Our imagery is orthorectified/RPC overhead where that abstraction is a poor fit and adds moving parts. We instead get metric grounding "for free" from LiDAR labels. We **borrow** Metric3D's *virtual-normal loss* idea (§2.5).
- **vs Depth Pro:** We want its **boundary sharpness**, not its pipeline. We port its **multi-scale gradient + Laplacian boundary losses and edge non-max-suppression** onto the DA2-L DPT head (§2.5). This gets ~80% of the edge benefit at ~0% of the migration cost.
- **vs HTC-DC Net wholesale:** Its **head-tail-cut classification-regression** is the right medicine for the documented *tall-structure underestimation* problem, but we keep DA2's superior pretrained encoder. Adopt the **head + distribution constraints**, not the whole repo.

### 2.4 Fine-tuning strategy: **full fine-tune with differential LRs** (primary) + **LoRA** (fallback/ablation)

- **Differential learning rates** (official DA2 metric recipe): encoder `5e-6`, decoder/head `5e-5` (10×). Rationale: preserve DINOv2's general features, let the DPT head adapt fast. [DA2 metric README]
- **Full FT is affordable** on 24 GB for Large at 518² with AMP (fp16/bf16) + gradient checkpointing + batch 4–8 and grad-accum. This is the strongest-accuracy option and our default.
- **LoRA/DoRA as the fallback and as the vehicle if we ever test Giant.** Evidence: depth-aware/dynamic LoRA (rank grows with layer depth) matches full-FT quality at ~35% less compute for dense prediction. Trade-off: LoRA slightly caps peak accuracy vs full-FT on a big domain shift, but de-risks catastrophic forgetting and lets us A/B backbones cheaply. **Chosen role:** experiment lever + Giant enabler, not the primary path, because we *can* afford full-FT on Large and want its ceiling.

### 2.5 Loss design (this is where accuracy is won or lost)

Compose four terms; each earns its place:

1. **SiLog (scale-invariant log)** — proven backbone loss for depth/height (Sat3R, DA2, HTC). Keep. Handles the relative-mode case gracefully.
2. **Metric L1/L2 on nDSM (meters)** — because we *have* metric labels; this is what turns the network metric and lets us drop `calibration.py`.
3. **Multi-scale gradient + Laplacian (edge) losses + edge NMS** — ported from Depth Pro; directly produces the crisp vertical building faces the mesh needs. This is the highest-leverage *quality* term.
4. **Long-tail correction (HTC-style)** — reweight/relabel so abundant ground pixels don't drown tall-building gradients. Fixes systematic height underestimation.
- *Optionally* a **virtual-normal / surface-normal** term (Metric3D idea) for smoother roofs and planar roads.
- **Trade-off noted:** more loss terms = more hyperparameters and tuning risk. We stage them (SiLog+L1 first, add edge, add long-tail) so each addition is measured against the §1 harness rather than stacked blindly.
- **Lean-first, with the VRAM myth corrected:** start with **single-scale Sobel gradient + Smooth-L1** — it delivers ~90% of the edge benefit and is one fewer thing to debug; only add Laplacian/multi-scale if the boundary-F metric stalls. A common worry is that a multi-scale/Laplacian boundary loss will OOM a 24 GB card — it won't: these losses operate on the **single-channel predicted depth map** (plus a couple of cheap downsampled copies), whose memory is negligible beside the ViT-L backbone activations. The real VRAM levers are **gradient checkpointing + batch size + grad-accum on the backbone**, not the loss. So we go lean for *debuggability*, not because the fancy loss is dangerous.

### 2.6 Data & training pipeline for GAMUS
- **GAMUS:** ~11,507 co-registered RGB + LiDAR-nDSM tiles, **1024×1024**, five US cities (Oklahoma, DC, Philadelphia, Jacksonville, NYC). *(GSD is stated by the repo docs as ~0.33–0.5 m; the arXiv/dataloader value should be verified — see Risks §8. This number feeds the metric-scaling and the super-res decision, so pin it early.)*
- Train at 518² random-crops from the 1024² tiles (also yields free scale augmentation); evaluate at native tile res via tiling (§4).
- Standard aug: flips/rotations (nadir imagery is rotation-tolerant), photometric jitter, **radiometric stretch matching** the inference-time percentile stretch in `geospatial.py` so train/serve distributions align.
- Hold out one entire city (e.g., NYC) as an **unseen-geography test set** — the honest generalization signal.

### 2.7 Domain adaptation to Indian terrain (the ISRO reality, not a footnote)
GAMUS is **five US cities** — wide grid streets, uniform suburban roofs, low-density zoning. ISRO SAC targets **dense, irregular Indian morphology**: Old-Delhi alleys, tin/brick rooftops, Himalayan valleys, Guwahati/Kedarnath floodplains, and very different albedos. A US-only model *will* throw artifacts the moment a jury drops a Cartosat/Sentinel tile on it. Mitigations, in order of leverage:
1. **Heavy radiometric + spatial augmentation tuned to Indian conditions:** aggressive brightness/contrast/shadow jitter, sun-angle/shadow simulation, haze, albedo shifts, and scale jitter (many small dense structures). This closes the *radiometric* gap cheaply.
2. **Validate on real Indian tiles:** assemble a small qualitative set (Cartosat / Sentinel-2 / Bhuvan); measure where labels exist, eyeball otherwise. This is the credibility slide for the jury.
3. **Any in-domain data is gold:** even a handful of Indian RGB↔height pairs (or ISRO **CartoDEM**-derived nDSM) for a short fine-tune/validation beats pure augmentation.
- **Honest limit (put it on the slide, don't overclaim):** augmentation narrows the *radiometric* gap but **not** the *structural/morphology* gap. Only in-domain data or an explicit domain-adaptation step truly closes that — say so rather than pretending a US-trained model is India-ready.

### 2.8 Dataset strategy — keep GAMUS as the core, extend for terrain diversity
The brief *recommends* GAMUS but permits any open remote-sensing depth dataset, and it grades **accuracy stability across urban / sparse / hilly / forested**. GAMUS is US urban/suburban, so pure-GAMUS under-covers two of the four. **Decision: anchor on GAMUS, extend selectively — do not replace** (a swap forfeits the "recommended dataset" credibility, and no single set beats GAMUS across all four landscapes).

Three reframes make extending cheap:
- **Hilly is a DEM problem, not a model problem.** The net predicts *height above ground* (nDSM); the hill/valley itself comes from the DEM base (§2A), and SRTM is explicitly blessed by the brief. GAMUS never needs hills.
- **Canopy height *is* nDSM** → forest datasets are drop-in compatible with the same regression head.
- **Mixed GSD across sources *helps* scale robustness** once the model is metric (and GSD-aware), rather than hurting it.

| Dataset | Adds | Role | Caveat |
|---|---|---|---|
| **GAMUS** (recommended) | Urban/suburban structure — 11.5k tiles 1024², LiDAR nDSM, 5 US cities | **Core trainer** + jury credibility | US-only morphology |
| **Open-Canopy** | **Forested** — SPOT-6/7 satellite RGB + aerial-LiDAR canopy height, 1.5 m, 87k km² | Add a slice for canopy | French forests; 1.5 m GSD |
| **GBH (Global Building Height)** | **Sparse / global diversity** — PLANET optical + nDSM + footprints, 19 world cities | Optional slice; narrows India gap | ~3–5 m PLANET GSD |
| **DFC2019 / US3D** | True **satellite** (WorldView-3) + LiDAR, Jacksonville/Omaha | Optional — matches satellite-input domain | overlaps GAMUS |
| **CartoDEM** (ISRO/Bhuvan, 30 m) | India **DEM base + validation reference** (ISRO's own product) | Absolute anchor + eval, **not training** | 30 m; terrain RMSE ~1.2–1.8 m |

**Recommended mix:** GAMUS core → metric nDSM fine-tune; *if time*, blend small **Open-Canopy** (forested) + optional **GBH** (global/sparse) slices into the same nDSM head; absolute base = SRTM/Copernicus globally + **CartoDEM for India**; validate on a held-out GAMUS city **plus real Cartosat/Sentinel tiles** referenced against CartoDEM — the four-landscape credibility slide for the 50 % accuracy score.

**Fallback the brief explicitly allows:** if metric fine-tuning stalls, Milestone 2 ("map relative depth → absolute via low-res DEM / semantic priors / GCPs") sanctions a *relative-depth + smart-calibration* path. Our DEM anchor (§2A) already satisfies that milestone either way — so the metric fine-tune is the *upgrade*, not a required gamble.

### 2.9 Multi-dataset harmonization (how to combine sources without corrupting the model)
Mixing datasets fails if you ignore that their differences come in **two kinds**:
- **Harmonizable (plumbing):** GSD, image size/format, radiometry/sensor, value range. Fix by **resampling all sources to a common GSD** (and/or **conditioning the model on GSD** — feed meters/pixel like a camera intrinsic), per-dataset radiometric normalization + photometric augmentation, and forcing every label to **meters, height-above-ground**. After this, GAMUS / Open-Canopy / DFC2019 / GBH-nDSM become one shared target.
- **NOT harmonizable (task identity):** **label semantics.** Dense per-pixel height (GAMUS, Open-Canopy, DFC2019) vs building *footprints* (SpaceNet-4) is a different *task*, not a rescale. Footprint-only data **cannot** join the height-regression loss — no rescaling makes a polygon into a per-pixel height.

**Rule:** one backbone, **one height head** fed by all dense-height-in-meters sources; **SpaceNet-4 only ever attaches as a *separate* head** (footprint / off-nadir-angle) or is skipped entirely.

```
                    ┌─ Height head ← GAMUS, Open-Canopy, DFC2019, GBH   (P1, now)
   Shared backbone ─┤
                    └─ Footprint/angle head ← SpaceNet-4                (v2 only)
```

**Sequencing:** train on **GAMUS alone first** (one consistent parameter set) → get the metric model working → *then* add a harmonized Open-Canopy/GBH slice. Multi-dataset blending is a classic bug source; earn it after the baseline. Balance batches so a large set doesn't drown a small forested slice. Full off-nadir handling: see **`OFF_NADIR_SOLUTIONS.md`**.

---

## 2A. nDSM vs absolute DSM — the base-terrain anchor (must-fix for flood sim)

**The trap (a real geospatial bug, not a nitpick):** GAMUS labels are **nDSM** — *height above the immediate local ground*. A 15 m building reads `15` whether it sits in Mumbai (~5 m above sea level) or Shimla (~2200 m). nDSM carries **no absolute elevation and no natural terrain slope** — riverbanks and valleys are flattened to 0. If we make the network predict nDSM and then *delete* the base-terrain model, two things break hard:
- **3D relief collapses:** mountains and valleys render flat, with only buildings poking up.
- **Flood simulation becomes physically meaningless:** ground is 0 everywhere, so a waterline either floods nothing or drowns the entire tile at once. You cannot model a river rising through a sloped floodplain — which is the whole ISRO disaster use-case.

**The formulation to keep (two-tier):**

$$\text{Elevation}_{\text{absolute}}(x,y) = \underbrace{DTM_{\text{base}}(x,y)}_{\text{real ground datum}} + \underbrace{\hat{h}_{\text{nDSM}}(x,y)}_{\text{AI-predicted metric height}}$$

- **The AI predicts only $\hat h_{\text{nDSM}}$** (object/canopy height in meters) — exactly what the GAMUS fine-tune gives us.
- **$DTM_{\text{base}}$ comes from a *real* source, not morphology guesswork:** for georeferenced tiles, fetch a coarse public DEM — **Copernicus GLO-30** or **SRTM 30 m** (or ISRO **CartoDEM** for Indian AOIs) — for the tile footprint via its CRS/affine bounds and upsample to the grid. For non-georeferenced PNGs, fall back to a flat base (relative mode); no absolute datum is recoverable there.

**So `calibration.py` is *repurposed*, not deleted:** it stops *inventing* the ground (morphological min-filter) and the scale (percentile α), and instead **composes a real DEM base with the network's metric nDSM**. Less code, more-correct math. Trade-off: adds a DEM fetch/dependency (cache tiles; bundle an offline DEM for demo AOIs so the jury demo never hits the network) — a fair price, and the only way the flood tool is honest.

---

## 3. Thread B — Off-nadir / tilted satellite imagery

### 3.1 Why tilt breaks a nadir monocular model
- **Roof–footprint offset:** at off-nadir angle θ, a building's roof is displaced from its footprint by `Δ = height · tan(θ)`. A nadir-trained height regressor has never seen facades and mis-attributes this parallax.
- **Facade visibility & occlusion:** oblique views reveal walls and cast long occlusions; nadir training has ~none of this.
- **Sensor geometry:** pushbroom satellites use **RPC/RPB** rational-polynomial models, not a pinhole center — the reason generic depth models (and Metric3D/UniDepth's canonical camera) fail on raw satellite frames, per Sat3R.

### 3.2 Approaches considered
| Approach | What it does | Verdict |
|---|---|---|
| **Off-nadir-angle + roof-offset auxiliary heads (MLS-BRN)** ⏸️ *v2 only* | Predict θ and offset direction *from the image* (metadata-free); relate roof/footprint/height | SOTA for *single* off-nadir images, but needs per-building vector annotations GAMUS lacks → 1–2 week build |
| **Synthetic oblique augmentation from nadir nDSM** ❌ **Rejected** | Warp / re-render GAMUS (RGB+nDSM) as if tilted | **Corrupts the label**: a 2D warp breaks the height↔image correspondence (parallax is height-dependent); a 3D re-render fabricates wall texture + occluded pixels the ortho doesn't have. Trains the model on fiction. |
| **Real off-nadir dataset (SpaceNet-4)** ⏸️ *v2 if truly needed* | Fine-tune on genuine 7°–54° off-nadir Atlanta tiles | The *honest* way to get oblique signal — real (image, height) pairs, no synthesis. Different label format; US; extra effort |
| **Detect + flag oblique inputs** ✅ *selected* | Lightweight image-based tilt cue → warn + degrade gracefully | Metadata-free, no training corruption, honest to the jury |
| Full generative photogrammetry ("Orbit to Ground") | Diffusion reconstruction from *extreme* off-nadir | Impressive but heavy/over-scoped for our app now — watch, don't build |
| Orthorectification / RPC rectification | ❌ Unavailable | We have **no RPC metadata**; a plain affine GeoTIFF maps to a flat plane and does **not** remove building lean. Conditional path only if RPC tiles ever appear |

### 3.3 Decision — off-nadir is **scoped, detected, and flagged**, never synthesized
**Two hard constraints kill the obvious fixes:**
- **No RPC metadata → can't straighten.** A plain affine/CRS GeoTIFF maps to a *flat* plane, which does **not** remove building lean (parallax is baked into the pixels; true de-leaning needs the sensor model + a DSM). So orthorectification is out.
- **Can't synthesize oblique training data from nadir nDSM → can't teach it either.** A 2D warp of (RGB, nDSM) breaks the height↔image correspondence because real tilt displaces tall objects *in proportion to their height*; a 3D re-render fabricates wall textures and occluded pixels the top-down ortho never captured. Either way the label becomes fiction and the model is **corrupted**. **We do not fake off-nadir data.**

**What we actually do (metadata-free, honest):**
1. **Treat DepthWizard as a nadir instrument.** Defensible — DSM/mapping tasking is overwhelmingly near-nadir, and GAMUS is near-nadir.
2. **Detect obliqueness at inference** with a lightweight image-based cue and **flag it** — "⚠ oblique input — reduced accuracy" — degrading gracefully instead of pretending to correct.
3. **Keep augmentation *appearance-only*** (brightness/contrast/shadow/haze for the India gap §2.7) plus flips / 90° rotations / scale — these preserve the image↔nDSM correspondence. **Tilt / perspective warps are explicitly excluded.**

**If genuine off-nadir support is ever required (v2):** fine-tune on a **real** off-nadir dataset (**SpaceNet-4**, 7°–54° off-nadir) — real (image, height) pairs, no synthesis — or add the image-based **MLS-BRN** angle/offset heads (needs annotations GAMUS lacks). Not before a demo.

**Honest limit:** a single oblique image can't recover a top-down DSM (occluded facades). Goal = **tolerate + flag**, and state off-nadir as a **known limitation on the slide** rather than ship a half-baked correction. **Priority:** below Thread A.

---

## 4. Thread C — Super-resolution the dataset (the "enhance for an edge" idea)

### 4.1 The hypothesis and the honest evidence
Idea: upscale GAMUS tiles (Real-ESRGAN / SwinIR / diffusion SR) to gain detail and thus accuracy.

**Evidence is genuinely mixed and often *negative* for metric tasks:**
- Downstream-benchmark studies: SR gains for detection/segmentation are typically **~4–6% F1 at best, with frequent regressions**; **transformer/neural-operator SR** helps more than **diffusion SR**, and there are "many cases where super-resolved performance is *lower* than the low-res original."
- For **depth/height specifically**, super-resolving *inputs* is risky: SR **hallucinates high-frequency texture**, and a *metric* model converts fake texture into **fake elevation**. Guided-depth-SR literature warns of **cascading errors** from upsampling.
- Super-resolving *labels* (nDSM) is worse — you'd be inventing LiDAR detail that doesn't exist.

### 4.2 The stronger alternative: native-resolution **tile-based inference**
The detail we actually want comes from **running the model at full resolution via tiling/patch refinement**, not from inventing pixels:
- **PatchFusion** (tile-based, on ZoeDepth): **+17–29% RMSE** on high-res benchmarks.
- **"One Look is Enough" / PRO** (ICCV 2025): single-pass patch refinement that removes the grid-boundary seams older tiling had — ideal for our 1024² tiles.
- This is **truthful detail** (real pixels, real LiDAR), no hallucination, and it composes with the metric fine-tune.
- **Latency caveat (delivery-critical):** tiling a 1024² image with overlapping 518² patches is **~9 ViT-L forward passes + blending ≈ 3.5–5 s**, *not* sub-second. So tiling is an **offline / export-quality** path, never the interactive one. Ship a **Live Mode** (single 518² resized pass, ~350 ms) for sliders and upload, and gate tiling behind a **[High-Precision GIS Tile Mode]** toggle used for export. Snappy demo, full accuracy on demand.

### 4.3 Decision & trade-offs
**Reject SR as a foundational step; adopt tile-based high-res inference instead.** Keep SR only as a **gated ablation**: *if* we ever ingest genuinely low-res sources (e.g., 10 m Sentinel-2) where there's no detail to recover, a **transformer SR front-end** may help — but it must be proven on the §1 harness against a no-SR baseline before shipping, precisely because it can hallucinate elevation. Trade-off: tiling costs more inference time per image (N patches) vs one SR pass, but buys accuracy honestly and avoids a metric-integrity risk we can't easily detect at serve time.

---

## 5. Thread D — The "melted-tent" mesh render artifact

### 5.1 Root-cause diagnosis (from the screenshot: EPSG:32617 metric urban tile, smooth mesh)
The buildings render as spiky tents with texture smeared down their flanks. **Three compounding causes:**
1. **Topology:** a continuous `PlaneGeometry` displaced by a heightmap **cannot represent a vertical wall** — a roof edge becomes a steep *diagonal* triangle strip (a ramp), not a face. Hence "tents."
2. **Depth softness:** ViT-S predicts **blurry discontinuities**, so even the roof edge is a *gradient*, widening the ramp. (Thread A's edge losses fix this at the source.)
3. **Texturing:** one nadir RGB texture is **UV-draped top-down** and **stretches** across the diagonal ramp (the roof's few edge pixels smear down the whole "wall"). This is the classic steep-slope texture-stretch problem.

The **Voxel mode looks right** because quantized blocks are extruded prisms — they *have* vertical faces and top-projected texels.

### 5.2 Fix ladder (cheapest → deepest) and trade-offs
| Fix | Effort | Effect | Trade-off |
|---|---|---|---|
| **Triplanar texture mapping** in the terrain shader ✅ *quick win* | Low (shader) | Kills the smeared "curtain" on steep faces immediately | ~2–3× texture samples (perf); still ramps, not walls — but *looks* far cleaner |
| **Edge-aware depth sharpening** (bilateral/guided filter, or upstream Thread-A edge loss) | Low–Med | Narrows ramps → steeper, crisper buildings | Over-sharpening can stair-step ground; tune |
| **Vertical "skirt" geometry** at detected depth discontinuities | Med | Inserts real vertical quads where Δheight is large → actual walls | Needs discontinuity detection + mesh surgery; more triangles |
| **LOD1 building extrusion** (segment footprints, extrude by median nDSM) | High | Crisp prismatic buildings on flat ground — the "proper" city look | Needs a footprint segmenter; changes the render model; separate ground vs buildings |
| Adaptive tessellation / triplanar displacement | High | GPU-generated extruded faces on steep heightfield regions | Complexity/perf; overkill vs skirts for our scale |

### 5.3 Decision
**Ship the quick pair first — triplanar texturing + edge-aware sharpening** (a day's work, no model retrain, immediate visible payoff). **Then, after the Thread-A metric model lands** (its edge losses make discontinuities detectable), **add skirt geometry**. Treat **LOD1 extrusion** as a stretch/"wow" milestone tied to a footprint segmenter. The user said skip if too deep — so triplanar+sharpen is the committed scope; skirts/LOD1 are optional upgrades. Note the honest limitation the user already intuits: from a **single nadir image you cannot *see* the walls**, so any wall is *synthesized* from the height jump — we make it look correct, we don't recover true facade texture.

---

## 6. Cross-cutting: the evaluation harness (do this first)
Everything above is unfalsifiable without numbers. Build a **`backend/evaluation/` harness** that: loads the GAMUS held-out split, runs the current pipeline and each candidate, and reports the §1 metric table (MAE/RMSE/δ1/boundary-F/long-tail bias) to a CSV + markdown. This converts "looks better" into "-1.7 m RMSE, +0.09 boundary-F," which is how the master plan gates every phase. `validation.py` already has RMSE/δ primitives to build on.

---

## 6A. Current implementation status — code audit

Grounded against the repo on this branch. **Headline: of the recommended stack, ~nothing is built yet, and the metric signal is actively discarded.**

| Technique / insight | Status | Evidence (file:line) | What's missing |
|---|---|---|---|
| Metric nDSM regression (the #1 win) | ❌ **Actively prevented** | target→[0,1] `dataset_gamus.py:187`; pred+target norm `train_gamus_full.py:299–303`; inference norm `depth_estimator.py:94`; loss clamp `losses.py:47` | un-normalize end-to-end; regress meters; repurpose calibration |
| Backbone → DA V2 **Large** | ❌ Still ViT-**S** | `config.py:11` (`…V2-Small-hf`) | swap `MODEL_ID`; re-tune batch for 24 GB |
| SiLog + gradient loss | 🟡 **Partial** | `losses.py` StableSILog + Sobel `GradientMatchingLoss` | Laplacian/multi-scale + edge NMS (optional) |
| Long-tail (HTC) term | ❌ | — | head-tail-cut reweighting |
| **Absolute DSM / DEM base-terrain** (flood-critical) | ❌ | `calibration.py:1–5` "no real SRTM" — statistical prior | fetch Copernicus/SRTM DEM; two-tier DSM (§2A) |
| Off-nadir handling | ❌ | no RPC in `geospatial.py`; aug = flip/rot90 only `dataset_gamus.py:110–122` | projective aug + RPC ortho (§3.3) |
| Eval harness / benchmark | ❌ | no `backend/evaluation/`; `validation.py` = helpers only | held-out benchmark runner |
| Super-resolution | ✅ **Correctly absent** | — | keep out; add tiling instead |
| Triplanar / mesh "melted-tent" fix | ❌ | `TerrainCanvas.tsx` standard material; `onBeforeCompile` = contour lines only | triplanar shader; skirt geometry |
| Tile-based hi-res inference (Live vs Export split) | ❌ | — | offline tiling path + fast Live Mode |

---

## 7. Consolidated technique-selection matrix

| Technique | Status | One-line rationale |
|---|---|---|
| Metric nDSM fine-tune (repurpose, not drop, calibration) | ✅ **Selected — P1, top priority** | Biggest measured win (Sat3R −38% MAE); we have LiDAR GT |
| DEM base-terrain anchor (two-tier DSM) | ✅ **Selected — flood-critical** | nDSM has no sea-level datum; keeps flood sim honest (§2A) |
| Extend GAMUS with Open-Canopy / GBH slices | ✅ Selected (time-permitting) | Covers eval's forested/sparse landscapes; same nDSM head (§2.8) |
| CartoDEM as India DEM base + validation reference | ✅ Selected | ISRO's own product; anchors + validates India (§2.8) |
| India domain augmentation + real-tile validation | ✅ Selected | Closes radiometric gap to ISRO tiles; honest re: morphology (§2.7) |
| Backbone → DA V2 **Large** | ✅ Selected | Accuracy/effort sweet spot on 24 GB |
| Differential-LR full fine-tune | ✅ Selected (primary) | Official recipe; affordable locally |
| LoRA/DoRA | ⏸️ Selected as fallback/ablation | Cheap A/B + Giant enabler; slight ceiling cost |
| Depth Pro boundary losses (grad/Laplacian/NMS) | ✅ Selected | Direct fix for soft edges → crisp walls |
| HTC head-tail-cut long-tail term | ✅ Selected | Fixes tall-structure underestimation |
| Virtual-normal loss | ⏸️ Optional | Smoother roofs/roads; add if roofs noisy |
| Tile-based high-res inference (PatchFusion/PRO) | ✅ Selected — P2, **offline/export only** | Honest detail; +17–29% RMSE — but ~4–5 s, so keep fast Live Mode |
| Off-nadir: detect + flag oblique inputs | ✅ Selected — P4 | Metadata-free, honest, no training corruption |
| Synthetic oblique augmentation from nadir nDSM | ❌ **Rejected** | Corrupts labels (warp) / fabricates pixels (re-render) |
| Real off-nadir data (SpaceNet-4) or MLS-BRN heads | ⏸️ v2 only | The honest way to get oblique signal; needs real data/annotations |
| RPC/ortho rectification | ❌ Unavailable | No RPC metadata; affine GeoTIFF can't remove lean |
| Triplanar texturing + edge sharpen | ✅ Selected — P3 (quick) | Immediate mesh-artifact relief |
| Skirt geometry / LOD1 extrusion | ⏸️ Stretch | "Proper" walls; needs segmenter/mesh surgery |
| Backbone → **Giant** | ❌ Deferred | ~3× cost, marginal aerial gain post-FT |
| Metric3D v2 / UniDepth as base | ❌ Rejected as base | Canonical-camera assumption ill-fit to overhead/RPC |
| Marigold diffusion base | ❌ Rejected | Too slow for a <1 s app; affine-invariant |
| **Super-resolving GAMUS to train** | ❌ Rejected as foundation | Mixed evidence + hallucinated elevation risk; tiling is the honest alternative |

---

## 8. Risks & open questions
- **GAMUS GSD must be verified** (repo says ~0.33–0.5 m; some sources cite coarser DFC values). It scales the metric target and decides whether *any* SR is even arguable. Pin from the `EarthNets/RSI-MMSegmentation` dataloader before P1.
- **Label noise / long tail:** LiDAR nDSM has registration error and a heavy ground "head." The HTC term and a robust (Huber) loss mitigate; monitor tall-building residuals specifically.
- **Domain gap GAMUS(US) → ISRO(India):** hold out a city, and plan a small India-tile qualitative check. Fine-tuning may overfit US urban morphology.
- **Relative-mode inputs** (non-georeferenced PNGs) still need a sensible scale; keep a lightweight relative path even after metric FT.
- **Preprint freshness:** several sources are 2025–2026 preprints (Sat3R, PRO, TSE-Net); treat their exact numbers as directional, not gospel.
- **Perf budget:** triplanar + tiling both cost GPU/time; keep the 60 FPS render and <1 s inference targets in view.

### 8.1 Loose ends found in the code audit (fix before re-training)
- **Backbone still ViT-S** (`config.py:11`) — the "→ Large" move hasn't been made; the training script also loads the Small `MODEL_ID`.
- **`CombinedLoss` call is out of sync with its definition:** `train_gamus_full.py:267` calls `CombinedLoss(alpha=0.5)`, but `losses.py:93` defines `__init__(alpha_silog, alpha_l1, alpha_grad)` — no `alpha` kwarg. Against the current `losses.py` this raises `TypeError`; the two files have drifted (archived training logs likely predate this `losses.py`). Reconcile before any fresh run.
- **Training script is sized for the old RTX 4060 8 GB laptop** (`train_gamus_full.py:8` — "batch 8 → 3.18 GB … RTX 4060"), not the 24 GB 4500 Ada. Leaves ~2/3 of VRAM idle; re-tune for Large + larger effective batch.
- **`best_model.pth` was trained in normalized [0,1] space** → it is a *relative-depth* domain adaptation, not a metric model. Don't mistake its existence for "metric is done."

---

## 9. References (primary sources used)
- Sat3R: Satellite DSM Reconstruction via RPC-Aware Depth Fine-tuning — https://arxiv.org/abs/2605.07264
- Depth Anything V2 — https://arxiv.org/abs/2406.09414 · metric recipe: https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/README.md
- Depth Pro: Sharp Monocular Metric Depth in Less Than a Second — https://arxiv.org/pdf/2410.02073
- Metric3D v2 — https://arxiv.org/abs/2404.15506
- HTC-DC Net: Monocular Height Estimation from Single Remote Sensing Images — https://arxiv.org/abs/2309.16486
- TSE-Net: Semi-supervised Monocular Height Estimation — https://arxiv.org/abs/2511.13552
- Enhancing Monocular Height Estimation via Weak Supervision from Imperfect Labels — https://arxiv.org/pdf/2506.02534
- THE Benchmark: Transferable Representation Learning for Monocular Height Estimation — https://arxiv.org/pdf/2112.14985
- GAMUS: A Geometry-aware Multi-modal Semantic Segmentation Benchmark — https://arxiv.org/abs/2305.14914 · code: https://github.com/EarthNets/RSI-MMSegmentation
- MLS-BRN: 3D Building Reconstruction from Monocular RS Images with Multi-level Supervisions — https://arxiv.org/html/2404.04823v1
- 3D Reconstruction of Buildings Using Single Off-Nadir Satellite Image — https://www.mdpi.com/2072-4292/13/21/4434/htm
- From Orbit to Ground: Generative City Photogrammetry from Extreme Off-Nadir — https://arxiv.org/html/2512.07527v1
- SpaceNet Off-Nadir dataset — https://medium.com/the-downlinq/introducing-the-spacenet-off-nadir-imagery-and-buildings-dataset-e4a3c1cb4ce3
- Benchmarking Super-Resolution via Downstream Task Integration — https://arxiv.org/pdf/2605.00310
- Data Augmentation Approaches for Satellite Image Super-Resolution — https://isprs-annals.copernicus.org/articles/IV-2-W7/47/2019/
- PatchFusion: Tile-Based High-Resolution Monocular Metric Depth — https://arxiv.org/abs/2312.02284
- One Look is Enough (PRO): Seamless Patchwise Refinement — https://arxiv.org/abs/2503.22351
- Depth Pro edge/boundary + SharpNet / occlusion-boundary refinement — https://arxiv.org/pdf/1905.08598 · https://arxiv.org/pdf/2002.12730
- Dynamic LoRA Fine-Tuning of DINOv3 (depth-aware LoRA) — https://pmc.ncbi.nlm.nih.gov/articles/PMC13517550/
- Triplanar mapping / displacement for terrain texture-stretch — https://www.cs.cit.tum.de/en/cg/research/publications/2020/triplanar-displacement-mapping/ · https://code.tutsplus.com/use-tri-planar-texture-mapping-for-better-terrain--gamedev-13821a
- RS3DBench: 3D Spatial Perception in Remote Sensing — https://arxiv.org/pdf/2509.18897
- Open-Canopy: Very High Resolution Forest Monitoring (SPOT + LiDAR canopy height) — https://arxiv.org/pdf/2407.09392
- Global Building Height (GBH) dataset — via HTC-DC Net refs — https://arxiv.org/pdf/2309.16486
- Data Fusion Contest 2019 (DFC2019 / US3D) — https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019
- ISPRS Vaihingen/Potsdam 2D/3D semantic labeling (9 cm nDSM) — https://www.isprs.org/education/benchmarks/UrbanSemLab/
- CartoDEM (ISRO/NRSC Bhuvan, 30 m Indian national DEM) — https://bhuvan-app3.nrsc.gov.in/data/download/tools/document/CartoDEMReadme_v1_u1_23082011.pdf
- A high-resolution canopy height model of the Earth (ETH, Sentinel-2 + GEDI) — https://www.nature.com/articles/s41559-023-02206-6
