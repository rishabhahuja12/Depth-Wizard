# DepthWizard — Voxel/Block DSM Render Mode

### Change plan for the coding agent (reasoning + decisions, no implementation)

---

## 0. TL;DR

Add a second render mode for the 3D view: a **quantized block/voxel extrusion** of the DSM (flat solid colors, hard edges, no smoothing — "Minecraft-like") as the **main** view, and demote the current photoreal smooth-displacement mesh to a **small fixed minimap** in the corner.

Key finding that shapes this whole plan: **the backend already returns everything needed, unchanged.** The mesh payload in `mesh_builder.py` includes `dsm_raw` alongside the 16-bit heightmap PNG (see the Step 3 sequence diagram in the original report — `Mesh-->>API: Mesh payload (16-bit Heightmap b64, Normal Map b64, Turbo b64, dsm_raw)`). `dsm_raw` is the actual float elevation grid at the already-decimated (≤512×512) resolution. That means the voxel view is a **frontend-only feature** — no backend changes, no new endpoint, no PNG-decoding precision loss. Everything below is written with that as the default plan, with a backend option flagged as optional Phase 2.

---

## 1. Naming this correctly (so the right prior art gets consulted)

What you're describing isn't SLAM (that's a robotics localization/mapping technique using sensor odometry over time — not applicable to a single static image) and it isn't really SAR (that's a radar imaging modality, not a rendering style). What you're picturing is a **quantized voxel heightfield extrusion** — the same family as:
- Minecraft-style block worlds
- CesiumJS / 3D Tiles "extruded building" city visualizations
- Voxel-mode point cloud viewers (Potree, CloudCompare)
- Classic hypsometric-tint relief maps, but stepped into discrete bands instead of a smooth gradient

Naming it this way matters because it's a genuinely different geometry strategy, not a shading tweak — see §3.

---

## 2. What stays, what changes

**Unchanged:**
- `geospatial.py`, `depth_estimator.py`, `calibration.py` — nothing about ingestion, inference, or DTM/nDSM calibration needs to change.
- `mesh_builder.py` — no changes required for the default plan (see §3.1).
- `TerrainCanvas.tsx` — reused as-is, just remounted smaller and non-interactive (see §3.6). Not deleted, not rewritten.

**New:**
- A voxel-grid builder (pure function, no rendering) that pools `dsm_raw` down to a block resolution and assigns each block a quantized color.
- A new renderer component for the block mesh.
- A small UI toggle/slider, and a second `<Canvas>` for the minimap.

---

## 3. Design decisions and reasoning

### 3.1 Where does the block grid get computed? → Frontend only, using `dsm_raw`

Two options existed:

- **Option A — Frontend pools the existing `dsm_raw` array.** Zero backend changes. `dsm_raw` is already at ≤512×512 (post `scipy.ndimage.zoom` decimation). Pooling that further down to, say, 48–96 cells per side is just block-averaging a matrix that's already in memory client-side.
- **Option B — New backend field/endpoint** that decimates straight from the full-resolution DSM to block resolution in one step, avoiding the "decimate twice" path (full-res → 512 → block-res).

**Recommendation: Option A, ship it first.** The two-stage decimation (server bilinear zoom to 512, then client box-average to ~64) is not meaningfully different from a single-stage decimation straight to ~64 for this use case — the whole point of the voxel view is to discard fine detail, so the second averaging pass is strictly a coarsening step, not a source of new artifacts. Option B is worth revisiting only if your coding agent actually sees banding/aliasing that's visibly different from what Option A produces — treat it as a Phase 2 optimization, not a prerequisite (see §4).

### 3.2 Where do the block colors come from? → Quantized Turbo bands, computed in JS, no image decoding

Two data sources were possible: sample the existing Turbo colormap PNG per block, or sample the optical RGB per block. Both require decoding a base64 image back into pixel data in the browser, which is extra work and (for the 16-bit heightmap specifically) would be lossy — browser Canvas `getImageData` only exposes 8-bit RGBA, so decoding the *heightmap* PNG that way would silently truncate the 16-bit precision you built specifically to avoid banding. This is a non-issue for elevation values, since you're pulling those from `dsm_raw` directly (§3.1), but it's a real trap if the coding agent instinctively reaches for the heightmap PNG.

**Recommendation:** Don't decode any image for coloring either. Reimplement a small posterized Turbo palette (7–10 hard-coded hex stops, same blue→cyan→green→yellow→red progression already used in `mesh_builder.py`'s colormap step) directly in JS, and map each block's normalized `dsm_raw` value to a band index → palette color. This:
- Keeps the new "solid color" voxel view visually consistent with the existing Turbo DSM legend already shown in the Inspect tab — same color language, discrete instead of continuous.
- Needs no new payload fields, no image decode, no backend change.

Treat "sample real optical RGB and quantize it" as a later, optional second color mode (toggle), not the default — it requires pooling the RGB texture too and tends to look muddier/less legible at block scale than a clean elevation-band palette.

**Band count:** expose as a slider, default around 7–8. Reasoning: below ~5 bands the elevation structure gets unreadable (everything looks like 2–3 flat plateaus); above ~12 it starts to look smooth again, which defeats the "solid color" ask. This is a judgment call — leave the exact default to whoever tunes it against real output, but keep the range roughly 5–12.

### 3.3 Geometry strategy → InstancedMesh, one instance per block

For the actual app (not a one-off demo), use `THREE.InstancedMesh` with a single shared `BoxGeometry`, one instance per grid cell, rather than one `Mesh` per cell. This is a single draw call regardless of block count, versus thousands of draw calls with individual meshes — matters for the "stable 60 FPS on standard laptops" bar already set for the smooth mode.

**One compatibility risk to flag for the coding agent up front:** per-instance color (`InstancedMesh.instanceColor` / `setColorAt`) was added in Three.js **r131**. Have the agent check the `three` version pinned in `package.json` before starting — if it's older (the report's own code snippets suggest the project may be on an older pin), either bump the dependency or fall back to a manual `InstancedBufferAttribute` for color with a small custom `ShaderMaterial`. This is a five-minute check that avoids a confusing runtime dead-end later.

### 3.4 Vertical anchoring — avoid floating or sunken blocks

Each block needs its **base** at a common ground plane (y = 0), not its center. Concretely: instance `position.y` should be `height / 2` and `scale.y` should be `height`, so the box's bottom face sits at y = 0 and it "grows upward" from the ground like the current smooth mesh does. Positioning at `position.y = height` (top-anchored) is the classic bug here and produces floating boxes. Worth a one-line comment in the code so nobody "fixes" it later.

Leave a small gap between adjacent blocks (scale the X/Z footprint to ~0.85–0.92 of the cell size) — this is what actually reads as "Minecraft" rather than "low-poly terrain." Without the gap, adjacent same-height blocks fuse into a single slab and the blockiness disappears visually even though the geometry is technically voxelized.

### 3.5 What happens to the existing surface features in voxel mode

- **Displacement/normal-map material** (`displacementMap`, `normalMap`, `displacementScale`, `normalScale` on `meshStandardMaterial`): these belong entirely to the smooth `PlaneGeometry` path and don't apply to boxes at all — box lighting comes for free from each face's own flat normal plus a directional light. The voxel component simply doesn't use this material setup. `mesh_builder.py`'s analytic normal-map generation (Step 3.4.3) keeps serving the smooth/minimap path only — no change needed there either.
- **Runtime contour-line shader injection** (`onBeforeCompile`, the `fwidth()`-based contour lines): this was written for a continuous `vTerrainWorldPos.y` varying smoothly across a triangulated plane. It doesn't have a natural equivalent on a box mesh, and — importantly — it's **redundant** in voxel mode: every color-band boundary between adjacent blocks *is* a contour line already, for free, as a side effect of §3.2. Recommendation: don't port the contour shader to voxel mode at all; it's replaced by the color banding, not lost.
- **Flood simulation water plane:** no change needed anywhere. It's an independent translucent horizontal plane positioned purely by a Y coordinate (`waterLevel`), so it sits correctly over either the smooth mesh or the voxel mesh without modification. Confirm this in testing, but don't budget engineering time for it.

### 3.6 Minimap — reuse `TerrainCanvas.tsx` as a second, smaller `<Canvas>`

Since the frontend is React + `@react-three/fiber`, the pragmatic path is **two independent `<Canvas>` elements**, not one WebGL context with manual scissor/viewport splitting (which is the "correct" answer in raw Three.js, but adds real complexity that isn't needed here):

- Keep `TerrainCanvas.tsx` essentially unchanged as the smooth-mesh renderer.
- Mount a second, small (~200×150px) instance of it, camera locked to a fixed non-interactive angle (disable orbit controls, or hardcode a top-down/three-quarter camera prop), positioned with CSS `position: absolute` in a UI corner over the main voxel view.
- The main `<Canvas>` now renders the new voxel component instead of `TerrainCanvas`.

Trade-off worth stating explicitly: two live WebGL contexts costs a bit more GPU/driver overhead than one canvas with scissor regions. At 200×150px this is negligible; it's the right trade for dev velocity in an R3F codebase. Revisit only if you see real perf problems on low-end devices.

### 3.7 File/component plan

- **New:** `frontend/src/lib/voxelize.ts` (or similar) — pure function, no rendering: takes the pooled `dsm_raw` grid, a target block resolution, and a band count, and returns per-block height + color. Kept separate from the renderer so it's independently testable and reusable if an optical-RGB color mode gets added later.
- **New:** `frontend/src/components/VoxelTerrain.tsx` — the `InstancedMesh` renderer consuming `voxelize.ts`'s output.
- **Modified:** wherever `TerrainCanvas` is currently mounted (likely a results/viewer container component not detailed in the original report) — mount both canvases, add a render-mode toggle.
- **UI placement:** put the Voxel/Smooth toggle inside the existing **"01 SURFACE"** tab rather than adding a 5th top-level tab. Reasoning: the existing tabs (Surface / Flood / Profile / Inspect) are organized around *what* you're looking at, not *how* it's rendered — a render-style switch belongs as a control within Surface, not as a new sibling concept.
- **Untouched:** `mesh_builder.py`, `calibration.py`, `depth_estimator.py`, `geospatial.py`, and `TerrainCanvas.tsx`'s internals.

### 3.8 Block resolution / perf budget

Default around **48×48 to 64×64** blocks (2,304–4,096 instances). Reasoning: dense enough to read as an actual city layout, coarse enough to look deliberately blocky rather than "smooth terrain that failed to render." Expose it as a slider from ~24 (very stylized) to ~96 (near-smooth but still hard-edged); `InstancedMesh` comfortably handles even the full 262,144-cell case, so this is purely an aesthetic knob, not a performance ceiling — no need to gate it behind a perf warning.

---

## 4. Suggested build order (for commit-sized milestones)

1. `voxelize.ts` — pure pooling + palette-quantization function, unit-tested against a synthetic height grid (no rendering yet).
2. `VoxelTerrain.tsx` — static `InstancedMesh` render, fixed band count, one directional + one ambient light, validated against one real sample dataset.
3. Minimap wiring — second `<Canvas>`, reused `TerrainCanvas`, fixed camera, CSS-positioned overlay.
4. UI — render-mode toggle + band-count slider in the Surface tab.
5. **Phase 2 (optional, only if needed):** backend single-stage low-res endpoint (§3.1, Option B); optical-RGB block color mode (§3.2).

---

## 5. Satellite imagery sources for testing

A couple of these I could confirm live just now; a couple I couldn't re-verify in this session due to a search outage on my end — flagged below so you double-check the exact current portal/access details before relying on them.

**Verified this session:**
- **SpaceNet** (AWS Open Data, `spacenet.ai`) — 30–50cm GSD WorldView-2/3 imagery over Vegas, Paris, Shanghai, Khartoum, Rio, with building footprint labels. Directly relevant to your use case. Worth calling out specifically: the related **Urban3D** dataset (also in this family) ships each RGB sample paired with a **ground-truth Digital Surface Model and Digital Terrain Model** — that's an actual reference DSM you can diff your reconstruction against, not just a nice-looking input image.
- **Copernicus Browser** (`browser.dataspace.copernicus.eu`) — free (account required) Sentinel-1/2/3/5P imagery, visualize-before-download UI. Sentinel-2 is 10m GSD, so it's a better fit for large-area/regional terrain testbeds than dense per-building urban scenes — good complement to SpaceNet's building-scale imagery.

**Not re-verified this session (confirm current access/URLs before relying on them):**
- **USGS EarthExplorer** (`earthexplorer.usgs.gov`) — free account, NAIP US aerial imagery (~0.6–1m) and Landsat.
- **Maxar Open Data Program** — openly licensed sub-meter imagery, typically real GeoTIFFs with proper CRS metadata, which makes it a strong match for your "Testbed A / metric mode" georeferenced pathway specifically.
- **ISRO Bhuvan** (`bhuvan.nrsc.gov.in`) — worth prioritizing given SIH26175 is an ISRO SAC challenge: a demo running on Bhuvan-sourced Cartosat/Resourcesat imagery is a direct, on-theme fit for judges. Some higher-resolution products on Bhuvan require registration/approval, so budget time for that if you go this route.

---

*This document is the plan only — no code has been written. Hand this to your coding agent as the spec; have it verify the `dsm_raw` payload shape (flat vs. row-major 2D array, and whether width/height come from `meshStats`) and the `three` package version before starting §3.3.*
