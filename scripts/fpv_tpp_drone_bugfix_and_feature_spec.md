# DepthWizard: FPV/TPP Drone HUD — Bugfix & Feature Expansion Spec

> **Target audience**: coding agent / developer picking up the drone flight work.
> **Status**: FPV drone mode has been implemented and is flyable (see attached HUD screenshot ref). This doc specifies **2 visual bugs to fix** and **4 features to add** on top of it.
> **Prereq reading**: `project_brief.md` and `reports/drone_and_navigation_mechanics_report.md` — this doc assumes the physics/camera model described there (YXZ euler, yaw-decoupled movement, now extended with thrust/drag/AGL per the blueprint in that report).

---

## 0. Context

The FPV cockpit HUD is live: pitch ladder, compass tape, airspeed/altitude/AGL telemetry boxes, mode switcher (`[1] MANUAL / [2] ORBIT / [3] TRANSECT`), link/battery/arm status. Two rendering artifacts are visible in the current build and need to be root-caused and fixed before new features are layered on top, since both features below (TPP mode, dual-mesh flight) will reuse this same HUD/camera-rig code.

**Note to agent**: the HUD/flight-controller component isn't named in the existing repo map docs (it postdates them) — locate it under `frontend/src/components/` (likely something like `FPVController.tsx`, `DroneHUD.tsx`, or a `FlightRig.tsx` + `Hud.tsx` pair) before starting. Confirm actual filenames and update this doc's references once found.

---

## 1. Bug Fixes (P0 — fix before building on top of this)

### 1.1 Flickering horizontal white/grey bar, left and right edges of HUD

**Symptom**: A solid horizontal bar renders across the screen at roughly the vertical center of the pitch ladder (around the `0°`/waterline area), stretching from the screen edge inward on both the left and right sides. It flickers rather than holding steady.

**Likely cause candidates** (verify against actual code before fixing):
1. The pitch-ladder "rung" component that renders the `0°`/horizon reference line is missing the same width/clamp styling applied to the `±10°`, `±20°` rungs (those render as short bracketed segments; this one may be defaulting to `width: 100%` or an unset max-width).
2. If the HUD is DOM/CSS overlay (not canvas-drawn), the flicker points to a value that's being recomputed and re-rendered via React state **every frame** (e.g. `setState` inside `useFrame`), causing the bracket width/position to thrash between two values each render — this is a common React Three Fiber anti-pattern. Fix by driving the rung's transform via a ref + imperative DOM/canvas update instead of state, or by memoizing/throttling the update.
3. Possible floating-point boundary flicker: a conditional like `if (pitch > 0)` toggling between two near-identical branches every frame when pitch hovers near zero. Add hysteresis or round the driving value before the conditional.

**Fix approach**: Instrument the horizon-rung element specifically, confirm whether it's a styling/clamp bug (static, one-line CSS fix) or a re-render/state-thrash bug (structural fix — move to imperative updates). Do not touch the other ladder rungs unless the same bug is found there too.

**Acceptance criteria**: Horizon line renders at a fixed, correctly-bracketed width matching the other rungs, with zero visible flicker held for 30+ seconds across pitch/roll input.

---

### 1.2 Diagonal/parabolic green line + floating dot, right of center

**Symptom**: A green line arcs from near the top of the screen down toward a green dot sitting roughly mid-right, with a second vertical green segment running from that dot straight up off the top of the frame. It reads as a stray marker/indicator line, not an intentional HUD element.

**Likely cause candidates**:
1. This is almost certainly a **world-space marker projected to screen space** (e.g. a home-point marker, POI/orbit-center indicator, or waypoint) using `vector.project(camera)` without guarding against the point being near/behind the camera's near-plane. When a projected point's `w` component is at or below the near-plane epsilon, the resulting NDC coordinates blow up or invert, producing exactly this kind of swinging/streaking line + wildly-offset dot as the drone moves.
2. Alternatively, it could be a compass-tape heading marker (e.g. "home" or "north" reference) whose screen-space wrap-around logic doesn't handle the 0°/360° boundary correctly, causing the indicator to sweep across the frame instead of clamping/wrapping.

**Fix approach**: Find whatever draws this marker (search for `.project(camera` or a "home point" / "POI" / compass-marker component). Add a behind-camera guard — if the marker's clip-space `w <= epsilon`, don't render it (or clamp it to the edge of the HUD with an off-screen indicator arrow instead, which is the standard FPS/flight-sim pattern). If it's the compass tape, fix the modulo/wrap math for the 0–360° boundary.

**Acceptance criteria**: No stray line/dot artifacts anywhere on screen through a full 360° yaw rotation and full range of pitch/altitude. If the marker is meant to be a real feature (e.g. "home point" indicator), it should render as a small on-screen icon only when actually in view, and as a clamped edge-of-screen arrow when off-screen — never as a streaking line.

---

## 2. New Feature: TPP (Third-Person) Drone Mode

Add a second camera mode — third-person chase view — using the **same flight physics/input pipeline** as FPV. Do not fork the controller; the physics/state (position, velocity, euler/roll) should be a single source of truth that both camera modes read from.

- **Mode toggle**: add a `[T] TPP` (or similar) button next to the existing camera mode indicator (`CAM: FPV NOSE [G]` in the current HUD) so the user can swap between FPV and TPP live, same as toggling gimbal mode.
- **Camera rig**: a chase camera offset behind and above the drone's body frame (e.g. `offset = (0, 2.5, -6)` in the drone's local space, `lookAt` the drone origin), with light smoothing/lerp on the camera follow so it doesn't feel rigid.
- **Visible drone model**: since the camera is now external, the drone needs an actual visible mesh in TPP mode (it doesn't need one in FPV, since the camera *is* the drone). Use a simple procedural quad-rotor placeholder (body box + 4 arms + rotor discs) if no model asset exists yet — don't block this on asset production.
- **Orientation**: the drone mesh's pitch/roll/yaw must visually reflect the same euler/roll state driving the FPV camera (thrust-pitch tilt, banking-into-turns roll, yaw) — this is the main payoff of TPP mode, since none of that banking/tilt is visible from inside the cockpit.

**Acceptance criteria**: Toggling between FPV and TPP mid-flight preserves position/velocity/orientation exactly (no jump/reset). TPP camera smoothly chases the drone through all maneuvers without clipping into terrain or the drone body.

---

## 3. New Feature: Fly Over Either Mesh (Smooth or Voxel)

Currently the flight rig is only mounted in the smooth-terrain scene (`TerrainCanvas.tsx`), not the voxel/box scene (`VoxelTerrain.tsx`). Add the option to fly over **either**.

- Simplest approach: reuse the existing `01 SURFACE` render-mode toggle (`Voxel` vs `Smooth`) as the single source of truth for which mesh is active, and mount the same flight-rig/HUD component into whichever scene is currently rendered — rather than building a separate toggle just for flight mode.
- **AGL collision consistency**: `getTerrainElevationAt()` currently samples `dsm_raw` directly (continuous values). In Voxel mode the *visual* surface is quantized into discrete elevation bands/blocks — if AGL collision keeps using the continuous raw value while the rendered surface is stepped, the drone will visibly float above or clip through the blocks it's flying over. When voxel mode is active, snap the sampled elevation to the same band quantization used by `VoxelTerrain.tsx`'s renderer (reuse its quantization function rather than re-deriving it) so collision matches what's on screen.
- Confirm the flight rig doesn't assume `TerrainCanvas`-specific scene structure (e.g. plane origin/scale conventions) — per the project brief both scenes share the same world-anchoring convention (`[0,0,0]` center, blocks base-anchored at `y=0`), so this should be a mounting/prop-passing change, not a physics rewrite.

**Acceptance criteria**: User can start or switch a flight session in either render mode and get correct, collision-consistent flight in both, with no need to leave flight mode to change terrain style (if feasible — otherwise, switching terrain style is allowed to pause/reset the flight session, but must not crash or desync the HUD).

---

## 4. New Feature: TPP Drone Size Slider

- Add a slider (in the TPP mode UI, only visible/relevant when TPP is active) that scales the visible drone model live: `droneGroup.scale.setScalar(sliderValue)`.
- Reasonable range: something like `0.5x`–`3x` of a sensible default — tune against the terrain scale so the drone stays visually legible at both extremes relative to buildings/terrain features.
- Scale should affect only the visual mesh, **not** the physics/collision radius, unless you also want bigger drones to be harder to fly close to buildings — flag this as a design decision rather than assuming; default to visual-only scaling unless told otherwise.

**Acceptance criteria**: Slider updates the drone's rendered size in real time with no physics side effects (unless collision-scaling is explicitly requested later).

---

## 5. New Feature: Terrain/Voxel Detail Quick Panel

Surface the existing voxel customization controls (discrete band count, block resolution, vertical exaggeration, contour interval — currently living inside the `01 SURFACE` sidebar tab per the project brief) via a **dedicated button on the main page/toolbar** that opens a floating panel, rather than requiring the user to navigate into the sidebar tab.

- Add a button (e.g. a gear/terrain icon) in the main header/toolbar that opens this panel as an overlay — should be reachable even while a flight session (FPV/TPP) is active, so users can tweak terrain coarseness mid-flight without exiting.
- **Don't duplicate the slider logic.** Factor the existing band/resolution/exaggeration/contour state out of `SettingsPanel.tsx` into a shared hook (e.g. `useTerrainSettings()`) that both the sidebar tab and this new quick panel read from and write to, so they stay in sync and there's one implementation to maintain.
- This panel is purely a UI-surfacing change — no new terrain-processing logic should be needed, since the underlying decimation/quantization already exists per the pipeline in `mesh_builder.py` / `voxelize.ts`.

**Acceptance criteria**: New panel button opens/closes cleanly, all controls reflect and mutate the same state as the existing `01 SURFACE` tab (changing one updates the other), and it's usable without interrupting an active flight session.

---

## 6. Non-Goals / Do Not Touch

Per the project brief's rules of engagement, this work stays entirely in `frontend/`:

- Do not touch `geospatial.py`, `depth_estimator.py`, or `calibration.py`.
- Do not change the `dsm_raw` payload shape (`number[][]`, row-major) — sample from it, don't reshape it.
- Maintain the 60 FPS budget: keep using `THREE.InstancedMesh` for the voxel drone/terrain repeats, dispose geometries/textures on unmount, and be mindful that adding a TPP drone mesh + a second live scene mount is extra render cost — profile after implementing.
- Keep the world-anchoring convention: origin at plane center, `+Y` vertical, voxel blocks base-anchored at `y = 0`.

---

## 7. Suggested File Touchpoints

| File | Relevance |
|---|---|
| `frontend/src/components/TerrainCanvas.tsx` | Smooth-mesh scene; current flight rig mount point |
| `frontend/src/components/VoxelTerrain.tsx` | Voxel-mesh scene; needs flight rig mounted here too (§3) |
| `frontend/src/components/SettingsPanel.tsx` | Source of existing band/resolution/exaggeration controls — factor into shared hook (§5) |
| `frontend/src/lib/voxelize.ts` | Quantization logic to reuse for AGL snapping in voxel mode (§3) |
| `frontend/src/App.tsx` | Main layout/router — add quick-panel button and TPP mode toggle here |
| *(unnamed FPV controller/HUD component — locate before starting)* | Bugs in §1, TPP camera rig in §2, dual-mesh mount in §3 |

---

## 8. QA / Acceptance Checklist

- [ ] Horizon-line flicker gone, held steady 30+ sec, both edges
- [ ] Stray diagonal green line/dot gone through full 360° yaw + altitude range
- [ ] `[T] TPP` mode toggles live without resetting position/velocity/orientation
- [ ] TPP chase camera never clips into terrain or the drone body
- [ ] Drone visual banking/pitch-tilt matches FPV's underlying state exactly
- [ ] Flight works in both Smooth and Voxel render modes
- [ ] AGL collision matches the visually rendered surface in Voxel mode (no floating/clipping)
- [ ] TPP drone size slider live-updates visual scale only
- [ ] New terrain quick-panel button opens/closes and stays in sync with `01 SURFACE` tab
- [ ] Quick panel usable mid-flight without breaking the HUD or dropping frame rate
- [ ] 60 FPS budget held on a standard laptop with all of the above active simultaneously
