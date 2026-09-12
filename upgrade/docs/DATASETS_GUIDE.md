# DepthWizard — Datasets Guide (Plain-English)

*A simple, complete explanation of every dataset we use (or chose not to use) to
train the height-estimation model, why each one matters, what it adds, and the
catches that come with it. Written to be readable without a remote-sensing
background. For the licensing/legal details see `DATASET_LICENSES.md`; for the
run commands see `../SETUP.md`.*

---

## 1. What are we actually building?

The project (SIH26175, for ISRO) turns a **single flat photo** taken from above
(a satellite or aerial image) into a **height map** — a picture where every pixel
says "how tall is the thing here" (a building, a tree, a hill). From that height
map we build a **3D world you can fly through**.

A normal camera photo has no height information — it's flat. So we teach a computer
model to **guess height from appearance**: shadows, roof shapes, textures, the size
of things. The only way it can learn that is by studying **many examples** where we
already know the true answer.

## 2. How a dataset teaches the model (the core idea)

Every training example is a **pair**:

- an **input**: the flat RGB photo, and
- the **answer**: a matching **height map** of the exact same patch of ground.

The model looks at millions of these pairs and slowly learns the pattern
"this kind of photo → this kind of height." That's it.

Two things make a dataset good for us:

1. **It has real pairs** (photo + true heights that line up perfectly). If the
   photo half is missing or can't be shared, the dataset is useless to us — no
   matter how good its heights are.
2. **It shows the model something new** (a kind of place it hasn't seen). Feeding
   it ten more datasets of the same city type doesn't help much.

### The four "places" we're graded on
The competition scores our accuracy across **four kinds of landscape**:

| Landscape | What it means |
|---|---|
| **Urban** | Cities — buildings, streets |
| **Sparse** | Thinly built areas, few structures |
| **Hilly** | Terrain that goes up and down a lot |
| **Forested** | Trees, canopy |

So our goal isn't "more data" — it's **cover all four kinds well**. That single
idea drives every choice below.

---

## 3. The datasets at a glance

| Dataset | What it gives | Status | Owns (why it's in the mix) |
|---|---|---|---|
| **GAMUS** (= the official **SIH2026** dataset) | Urban aerial pairs — the organizers' own data | ✅ **Core: powering the model & the baseline** | The real exam distribution + solid urban coverage |
| **Open-Canopy** | Forest/tree-height pairs | ✅ **Wired, next in line** | The **forested** landscape |
| **M4Heights** | City aerial pairs (Europe) | 🟡 **Planned (easy add)** | More urban depth; easiest to download |
| **DFC2023** | Building-height pairs, worldwide | 🟡 **Planned (diversity)** | **Sparse / hilly / non-Western** variety |
| **NL pair-builder** | Ultra-clean 0.5 m pairs (Netherlands) | 🟡 **Planned (only if needed)** | Highest-quality "truth" |
| **US3D / DFC2019** | Urban satellite pairs (USA) | ⏸️ **On hold (fallback)** | Nothing new — backup only |
| **GeoNRW** | Aerial pairs (one German region) | ⏸️ **On hold (code written)** | Redundant; kept ready |
| **SRTM** | Coarse ground-elevation map | 🔧 **Calibration, not training** | Turns "relative" height into real meters |
| **ISRO CartoDEM** | Coarse India elevation | 🔧 **Inference base (India)** | India ground level for the 3D output |
| **Google OB / AEEE** | India building heights (no photos) | 📏 **Eval only** | Measure how well we do in India |
| **GBH / GBA** | Global heights (no shareable photos) | ❌ **Dropped** | Can't train on it (no photo half) |
| **BuildingWorld** | LiDAR + 3D models | ❌ **Not applicable** | Different task |
| **Huawei BHE** | Claimed building heights | ❌ **Dropped** | Couldn't verify it exists/its license |

Legend: ✅ in use · 🟡 planned/gated · ⏸️ on hold · 🔧 used but not for training · 📏 for scoring only · ❌ not used.

---

## 4. Each dataset in detail

### ✅ GAMUS — the core trainer (this IS the official SIH2026 dataset)
- **What it is:** the aerial city images + height maps the competition organizers
  (ISRO SAC) provide — the SIH2026 dataset **is** GAMUS, not a separate set.
- **Why we use it:** it's our target distribution — the same kind of data we're tested
  on — and it's big, clean, and strong on **buildings** (the hardest structures). It's
  both the backbone of training and our honest scorecard.
- **Why it's important:** every improvement is measured against *this*. If a new
  dataset doesn't improve our score here, it wasn't worth adding.
- **What it contributes:** the model's core skill at reading rooftops/city layouts
  into heights, on exactly the distribution we're graded on.
- **Gaps/caveats:** it's mostly **urban** — it teaches little about forests or wild
  terrain. That's the whole reason we add the others (forest, sparse, hilly).
- **Status:** **used in full — the backbone of the pipeline.**

### ✅ Open-Canopy — the forest specialist
- **What it is:** aerial photos paired with **tree-canopy heights** (how tall the
  trees are), measured by laser (LiDAR).
- **Why we use it:** the exam grades us on **forested** land, and none of our other
  data teaches trees. This is the **only** dataset that owns that landscape.
- **What it contributes:** the model learns that leafy, textured green areas have a
  specific height signature — so it stops treating forests like flat ground.
- **Gaps/caveats:** it's French forests at a **coarser resolution (1.5 m/pixel)**
  and it's *trees, not buildings*. It's huge (~360 GB) so we only pull a slice.
- **Status:** **already wired; next to switch on in the blend.**

### 🟡 M4Heights — easy extra city data
- **What it is:** ~1 m aerial city photos + building heights across Netherlands,
  Germany, Switzerland (plus Sentinel-1/2 time series we don't need).
- **Why we use it:** more **urban** variety (European cities), deepening GAMUS.
- **What it contributes:** European city building diversity.
- **Gaps/caveats (revised after checking the real repo):** it's **not** the frictionless
  pull I first thought. The HF repo is **gated** (login + agreement), the data is in
  **`.zip` archives**, it's **mostly Sentinel (10 m, no-geolocation)**, and the 1 m aerial
  RGB + height we want is a **slice we must extract and unzip** before use. And it *deepens*
  urban rather than adding a new landscape — nice-to-have, not must-have.
- **Status:** **planned, but now behind Open-Canopy + DFC2023** given the extra handling.

### 🟡 DFC2023 — the diversity engine
- **What it is:** building-height pairs from **12 cities across 5 continents**.
- **Why we use it:** it's the **only** source that spans the whole world, including
  **non-Western, sparse, and hilly** areas. That directly targets the three landscape
  types our other data is weak on.
- **What it contributes:** generalization — the model sees many building styles and
  terrains, so it's less likely to fail on unfamiliar (e.g. Indian) scenes.
- **Gaps/caveats:** you must **register** to download it, only the training part has
  public answers, and **some tiles are slightly misaligned** (photo and height don't
  line up perfectly) — mild noise we may need to filter.
- **Status:** **planned — our main variety booster.**

### 🟡 NL pair-builder — the quality ceiling
- **What it is:** not a ready dataset — a **recipe**. The Netherlands publishes
  ultra-precise **0.5 m** laser elevation (AHN) *and* free aerial photos (PDOK)
  *and* bare-ground elevation. We combine them ourselves to build the cleanest
  possible photo+height pairs.
- **Why we use it:** laser "truth" at 0.5 m is the **best-quality answer key** we
  can get anywhere — better than any stereo-satellite estimate.
- **What it contributes:** a high-accuracy polish once the easier data is exhausted.
- **Gaps/caveats:** it's **the most work** — we have to line up the photos and the
  elevation ourselves (correct map projections, tile them, remove clouds/gaps). Same
  recipe later extends to the USA, Switzerland, Germany.
- **Status:** **planned — only if the easier datasets stop improving the score.**

### ⏸️ US3D / DFC2019 — kept as a backup
- **What it is:** high-resolution **satellite** photos of two US cities + heights.
- **Why it's on hold:** it only adds **more urban-US-satellite** data, which DFC2023
  already covers worldwide. Adding it would be breadth without new variety.
- **What it would contribute (if used):** a very clean US-satellite set — useful only
  as a **substitute** if DFC2023's misalignment turns out to be a problem.
- **Gaps/caveats:** manual download; its photos have **8 colour bands** (not plain
  RGB) so we must pick the right three; its file names don't match between photo and
  height (we handle both automatically, but they're reasons it's fiddly).
- **Status:** **on hold — fallback only.** The code already knows how to read it.

### ⏸️ GeoNRW — written but dormant
- **What it is:** aerial photos + laser elevation for one German region (NRW).
- **Why it's on hold:** it's **redundant** with the NL/US recipe, and it ships
  *surface* elevation (tree/roof tops **including the hill underneath**) rather than
  **height above the ground**. To use it we must subtract a separate bare-ground map
  first — and that bare-ground map isn't in the easy download.
- **What it contributes (if used):** one more clean European region.
- **Gaps/caveats:** the missing bare-ground step is the blocker; commercial-friendly
  licence is a plus.
- **Status:** **on hold. The code to read it and to do the subtraction is written
  and tested**, waiting only for the bare-ground data.

### 🔧 SRTM — not training, but critical
- **What it is:** a free, coarse (~30 m) worldwide map of ground elevation.
- **Why it matters:** our model naturally outputs **relative** heights ("this is
  higher than that") but not **real meters**. SRTM is the **ruler** we use at the end
  to convert relative heights into actual metres for georeferenced images — exactly
  the method the competition suggests.
- **Gaps/caveats:** it's coarse, so it fixes the **overall scale**, not fine detail.
  ⚠️ **Right now this step is faked** — the current calibration uses a rough guess,
  not real SRTM. Wiring in real SRTM is one of our biggest pending accuracy wins.
- **Status:** **used at inference (calibration), currently a placeholder.**

### 🔧 ISRO CartoDEM — India ground base
- **What it is:** ISRO's ~30 m elevation map of India.
- **Why it matters:** for Indian scenes, it gives the **base ground level** onto which
  we add our predicted building/tree heights to get true elevation for the 3D view.
- **Gaps/caveats:** far too coarse to see individual buildings — it's a **base layer
  only**, used when serving results, not for training.
- **Status:** **planned for the India inference path.**

### 📏 Google Open Buildings / AEEE GOBS — India scoring, not training
- **What they are:** building **footprints + estimated heights** for India (from
  Google), and an India dashboard built on top (AEEE).
- **Why they matter:** they let us **measure** how well our model does on **Indian**
  buildings, which none of our training data covers.
- **Gaps/caveats:** they come **without the original photos** (the imagery is
  private), so we **can't train** on them — only check our answers against them.
- **Status:** **eval-only track (measuring the India gap).**

### ❌ Dropped / not applicable
- **GBH / GlobalBuildingAtlas:** great global heights, but the matching photos are
  **private satellite imagery that can't be shared** — no photo half, so we can't
  train on it. (Same trap as Google's data.)
- **BuildingWorld:** laser scans + 3D models for a *different* task (building 3D
  reconstruction), not photo→height.
- **Huawei BHE:** we couldn't confirm it's actually downloadable or how it's
  licensed. Dropped.
- **ISPRS Potsdam/Vaihingen:** real pairs but only **two German towns** — too small
  and same-looking to teach anything new.

---

## 5. How they work together (the flow)

We add datasets **one at a time**, and each new one has to **actually improve the
score** on held-out data before we keep it. This "prove it or drop it" gate stops us
from piling on data that doesn't help.

```
Start:  GAMUS (urban core)
  └─► + Open-Canopy   (adds forests)        ── must beat the score, or revert
        └─► + M4Heights   (more cities, easy) ── must beat the score
              └─► + DFC2023  (worldwide variety) ── must beat the score
                    └─► NL pair-builder (top-quality polish) ── only if still needed
```

On hold the whole time (used only if something above fails): **US3D, GeoNRW,
and the US/Switzerland/Germany versions of the pair-builder.**

Separately, at the end of the pipeline (not training):
- **SRTM** converts relative heights to real metres,
- **CartoDEM** supplies India's base ground level,
- **Google OB / AEEE** tell us how we're doing in India.

---

## 6. Cross-cutting caveats (true for all the paired datasets)

These are the recurring "gotchas." The good news: our shared reader already handles
the mechanical ones automatically.

**Handled for us (in code):**
- **"No-data" holes and weird units** — some height files mark empty pixels with odd
  numbers (like 32767) or store heights in the wrong unit. Left alone, a single such
  pixel becomes a fake 32-km skyscraper and ruins training. We clean these
  automatically.
- **Mismatched file names** — one dataset calls a photo `..._RGB` and its height
  `..._AGL`. We match them anyway.
- **Extra colour bands** — satellite photos can have 8 bands; we pick the correct
  three for true colour.
- **Surface vs above-ground height** — for laser data we subtract the bare-ground
  map so "height" means *above the ground*, not *above sea level*.

**Verified structure notes (what the real downloads actually look like):**
- **Open-Canopy** ships as *per-year virtual rasters + a `geometries.geojson`*, **not**
  flat matched (RGB, height) tiles. Needs a pre-tiling step (window-read by geometry)
  before our reader can pair it. (Reader now globs recursively, which covers year-nested
  *tiles* if present.)
- **M4Heights** is **gated** (login + agreement), stored as **`.zip` archives** and is
  **Sentinel-1/2-heavy (10 m, "NOGEO")**; the 1 m aerial ortho is a *slice* to extract.
  So it's *not* a one-command drop-in — see the M4Heights note below.
- **US3D/DFC2019** RGB is **3-band pan-sharpened** (`JAX_004_006_RGB.tif`) — *not* 8-band,
  so no band-selection needed; but the AGL truth is **per-tile** (`JAX_004_AGL.tif`) while
  RGB has an extra image-id, so simple `_RGB`→`_AGL` remap won't map many-views-to-one —
  needs a stem rule that drops the image-id. (Adapter written as a fallback via `--us3d-root`.)
- **GeoNRW** uses per-city folders, **`.jp2`** RGB (`*_rgb.jp2`) + `*_dem.tif` (a **DSM**),
  so it needs `rgb_glob="*.jp2"` + the DSM→nDSM step. (On hold.)

**Real limits we can't code away (be honest about these):**
- **Folder layouts differ** — the exact folder/file names of each download must be
  confirmed by hand before a real run.
- **Domain gap to India** — almost all good open data is Western. The model may still
  struggle on Indian scenes; that's why DFC2023 (worldwide) and the India-eval track
  matter.
- **Different resolutions and dates** — datasets vary in sharpness (0.5 m to 3 m) and
  the photo and height can be captured on different days; we harmonize resolution, but
  perfect alignment isn't guaranteed.
- **Licences** — most are fine for a **non-commercial** ISRO project; a couple would
  need removing if the project ever went commercial. See `DATASET_LICENSES.md`.

---

## 7. Glossary (plain terms)

| Term | Plain meaning |
|---|---|
| **DSM** (Digital Surface Model) | Height of everything — tops of buildings, trees, ground. |
| **DTM / DEM** (Terrain / Elevation Model) | Height of the **bare ground** only (no buildings/trees). |
| **nDSM** ("normalized" DSM) | Height **above the ground** = DSM − DTM. What we actually want. |
| **Relative vs absolute height** | Relative = "higher/lower than"; absolute = actual metres. |
| **GSD** (Ground Sample Distance) | How much ground one pixel covers, e.g. 0.5 m/pixel = sharp. |
| **LiDAR** | Laser scanning that measures very accurate heights — the gold-standard "answer." |
| **Paired data** | A photo and its matching, lined-up height map — what training needs. |
| **RGB** | An ordinary colour photo (Red-Green-Blue). |
| **SRTM** | A free, coarse worldwide ground-elevation map. |
| **Blend / gating** | Adding datasets one by one, keeping each only if it improves the score. |

---

*Last updated: 2026-09. Companion docs: `DATASET_LICENSES.md` (legal/licence detail),
`../SETUP.md` (how to run training and the blend).*
