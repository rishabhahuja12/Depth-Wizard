# Off-Nadir / Tilted-Imagery — Solution Catalog

> Every approach considered for the "buildings seen from the side" problem, with mechanism, requirements, pros/cons, and verdict.
> Companion to `DEPTH_MODEL_RESEARCH_DUMP.md` §3. **Legend:** ✅ do now · ⏸️ v2 · ❌ rejected · 🔬 research-only.

---

## The problem (recap)
When the satellite views off-nadir (angle θ), tall buildings **lean**: the roof is displaced from its footprint by `Δ = height · tan(θ)`, facades/walls become visible, and long occlusions appear. A **nadir-trained** height model has never seen this and misreads the lean as a sloped ramp → wrong heights + the "melted-tent" render.

## Three hard constraints (why several obvious fixes are dead on arrival)
1. **No RPC camera metadata** → we cannot geometrically de-lean a tile (a plain affine/CRS GeoTIFF maps to a *flat* plane; the lean stays baked in).
2. **Can't synthesize oblique (RGB, nDSM) pairs from nadir data** → a 2D warp corrupts the label (parallax is height-dependent); a 3D re-render fabricates wall texture + occluded pixels the ortho never captured.
3. **Single-view occlusion is physical** → the far side of every building is hidden in one image; no method recovers it without another view.

---

## Solution catalog

| # | Solution | Mechanism | Needs | Pros | Cons | Verdict |
|---|---|---|---|---|---|---|
| 1 | **Detect + flag (nadir-only + warning)** | Lightweight image-based tilt cue → show "⚠ oblique — reduced accuracy" + degrade gracefully | Small tilt detector/heuristic | Metadata-free, honest, zero corruption, cheap, ships in days | Doesn't *improve* accuracy — only warns | ✅ **DO NOW** |
| 2 | Do nothing / silent degrade | Feed oblique straight in | — | Simplest | Silent wrong output; worse than flagging | ❌ Rejected |
| 3 | RPC orthorectification | Reproject to nadir via sensor model + DEM | **RPC (we lack)** + DTM | Clean geometric fix | Unavailable; bare-DTM ortho still smears tall buildings | ❌ Unavailable (conditional if RPC ever appears) |
| 4 | Homography/affine "straighten" without RPC | 2D warp using an estimated angle | Angle estimate | Cheap | **Wrong geometry** — same warp for all pixels, but parallax scales with height; only ~ok for low-relief/tiny tilt | ❌ Rejected for urban |
| 5 | Synthetic oblique augmentation from nadir nDSM | Warp/re-render GAMUS as if tilted | Nothing new | *Would* teach parallax if valid | **Corrupts labels / fabricates pixels** (constraint 2) | ❌ Rejected |
| 6 | Real off-nadir training data (SpaceNet-4) | Train on genuine 7°–54° imagery | SpaceNet-4 (footprints only, no dense height); US | Real oblique signal | Label-type mismatch → needs own head; US-only | ⏸️ v2 (separate head) |
| 7 | **MLS-BRN multi-task heads** (angle + roof-offset) | Predict θ and roof→footprint offset *from the image*, correct geometry | Per-building vector annotations GAMUS lacks | SOTA for single-image off-nadir | 1–2 week build; annotation pipeline | ⏸️ v2 (the "right" heavy fix) |
| 8 | Off-nadir **angle conditioning** of the height head | Feed predicted/known θ into the height model so it compensates | Height model must have *trained on oblique dense-height data* | Elegant, single model, no cascade | **Blocked**: we have no oblique dense-height labels → the model can't learn to use θ (circular) | ⏸️ v2 (blocked by data) |
| 9 | **Two-stage cascade: SpaceNet front-end → height model** (the proposed idea) | Stage 1 (SpaceNet) estimates footprint/angle; Stage 2 predicts height | SpaceNet model + a correction step | Intuitive; reuses off-nadir knowledge | Doesn't escape constraints; 2× latency; error propagation; correction needs height (circular) — see deep-dive | ⏸️ v2, and **joint > cascade** |
| 10 | Multi-view / stereo | Combine ≥2 looks | Multiple images | Truly recovers 3D + occluded sides | Out of scope — the brief is *single-view* | ❌ Out of scope |
| 11 | Shadow-based height cue | Infer height from shadow length + sun angle | Sun geometry + clean shadows | Metadata-light | Fragile; doesn't de-lean, only estimates height | 🔬 Research-only |

---

## Deep-dive: "put the input through SpaceNet first, then the other datasets"

**The instinct is good** — use SpaceNet-4's off-nadir knowledge as a front-end that "prepares" the image before the height model. In fact this is a re-derivation of the **MLS-BRN** idea (#7), just structured as two sequential models instead of two heads. But under our constraints it doesn't actually solve the problem:

1. **SpaceNet-4 outputs footprints/angle — not a corrected image and not a height.** So "passing the input through SpaceNet first" gives you *building masks* (and possibly an angle), **not a de-leaned tile**. There's no straightened image handed to Stage 2.
2. **Turning that into a correction needs the height you don't have yet.** To shift a leaning roof back over its footprint you need `Δ = height · tan(θ)` — i.e., the very height Stage 2 is supposed to predict. **Circular dependency.** MLS-BRN only breaks this circle by predicting angle, offset, and height *jointly with shared supervision* — not in a one-way cascade.
3. **The height model never trained on oblique dense-height data** (constraint 2). Even if Stage 1 handed it a perfect angle, Stage 2 wouldn't know how to use it, because we have no oblique (RGB, nDSM) pairs to teach that compensation.
4. **Practical costs:** two models = **~2× inference latency** (already tight for the <1 s Live Mode), and a sequential cascade **propagates Stage-1 errors** into Stage 2 with no way to correct back — a joint model is strictly more robust.
5. **It still can't recover occluded facades** (constraint 3).

**Verdict:** the idea points at the right *destination* (use off-nadir geometry knowledge), but the right *vehicle* is a **joint multi-task model** (#7) trained on **real** oblique data (#6) — not a sequential SpaceNet→height cascade. And all of that is **v2**: it needs annotations/data we don't have, and it fights the latency budget.

### Architecture: one height head; SpaceNet is a *separate* head (never a cascade)
All **dense height-in-meters** sources share ONE backbone + ONE height head. Footprint-only SpaceNet-4 can only ever be a *second* head — it is **not** mixed into the height loss, and it is **not** a sequential front-end.

```
  UNIFY  ✅  (what we do)                         CASCADE  ❌  (the rejected idea)
  ─────────────────────────                       ────────────────────────────────

                ┌─ Height head ◀── GAMUS          input ─▶ SpaceNet model ─▶ footprints / angle
                │   (metric nDSM,   Open-Canopy                                    │
   backbone ────┤    meters)        DFC2019                                        │  ⚠ correction needs
   (shared)     │                   GBH                                            │     HEIGHT (= Stage 2's
                │                                                                  ▼     own output) → circular
                └─ Footprint/angle head ◀── SpaceNet-4              Height model ─▶ height
                     (v2 only, separate loss)                       (never trained on oblique)
                                                                    = circular + 2× latency + error propagation
```

**Left** = unify all dense-height sets on one head; SpaceNet is an optional *second* head. **Right** = the sequential cascade, which breaks because Stage-2's correction needs the height Stage-2 hasn't computed yet, and Stage-2 never learned to use an angle anyway.

---

## What we actually ship

- **Now (P1 era):** **#1 — detect + flag.** Nadir-only instrument, honest "⚠ oblique" badge, graceful degradation. No corruption, no latency hit, no fake data.
- **v2 (only if oblique becomes a priority):** build the **joint MLS-BRN-style multi-task model** (#7), trained with **real** off-nadir data (#6, SpaceNet-4 as a *separate head*), optionally adding **angle conditioning** (#8) once real oblique dense-height labels exist.
- **Never:** synthetic oblique augmentation (#5), naive homography straightening (#4), or a one-way SpaceNet→height cascade (#9) as the "fix."

**One line:** *for the hackathon, off-nadir is a stated limitation we detect and flag; the real fix is a joint multi-task model on real oblique data, which is v2.*
