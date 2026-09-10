# DepthWizard: FPV Drone Mode — Implementation Plan

> **Status**: Planning doc, pre-implementation.
> **Scope**: A brand-new First-Person Drone flight mode, separate from the existing spectator/free-fly camera in `TerrainCanvas.tsx` / `VoxelTerrain.tsx`. The existing controls stay untouched.

---

## 1. The Core Question First: Do We Need to Download a Model?

**Short answer: No — not for v1.** Build the drone procedurally out of primitive Three.js meshes (boxes + cylinders), the same way `VoxelTerrain.tsx` already builds terrain out of `InstancedMesh` blocks. This is the right call for three reasons:

1. **Zero licensing/asset overhead.** No file to vet, no attribution file to maintain, nothing to fetch at build time.
2. **Full control over the rig.** The camera needs to sit at a precise "nose" offset, the props need to spin based on live throttle, and the arms need to be exact anchor points for the proximity-sensor rays. A procedural rig gives you named, code-addressable parts (`body`, `armFL`, `rotorFL`, `propFL`...). A downloaded mesh is an opaque blob you'd have to reverse-engineer.
3. **Fits the existing visual language.** DepthWizard already has a stylized, low-poly, Turbo-colormap aesthetic (`VoxelTerrain.tsx`). A photoreal drone model would look out of place; a clean procedural quadcopter matches it.

**If you want a higher-fidelity model later**, that's a Phase-7 stretch goal, not a blocker — see §7. The architecture below is built so a procedural rig and a loaded `.glb` are interchangeable behind one `<DroneModel />` component.

### Where to get a free model, if/when you want one
I checked current sources so this isn't stale advice:
- **[Poly Pizza](https://poly.pizza/search/Drone)** — dedicated, searchable "Drone" category, GLTF/GLB + FBX/OBJ, no login required. Many (not all) entries are CC0; check the license badge per-model before using — some Quaternius assets on there are CC-BY (attribution required).
- **[Kenney.nl](https://kenney.nl/assets)** — everything on Kenney is blanket **CC0**, no attribution needed, ever. Simplest legal path if you go this route. No dedicated "drone" pack, but their vehicle/space kits have close analogs and their CC0 blanket license means zero paperwork.
- Marketplaces like CGTrader/TurboSquid have "free" listings too, but licenses vary per-model and per-uploader — more friction, not recommended unless nothing else fits.

If you do load a `.glb`: use `@react-three/drei`'s `useGLTF`, drop the file in `frontend/public/models/drone.glb`, and keep it under a few hundred KB / low poly-count to stay inside the 60 FPS budget from the project rules.

---

## 2. Architecture: A New, Isolated Flight System

Do **not** extend `TerrainCanvas.tsx`'s camera controller. Build a parallel system so the existing spectator navigation (which the report already documents as "rock-solid") is never at risk of regression.

```
frontend/src/
├── components/
│   ├── drone/
│   │   ├── DroneFlightController.tsx   # physics integration, input → velocity/orientation, owns the drone's transform
│   │   ├── DroneModel.tsx              # procedural mesh: body/arms/rotors/spinning props/nav lights
│   │   ├── ProximitySensors.tsx        # 3 sensor rays (left/right/bottom), draws laser lines + feeds HUD
│   │   ├── DroneHUD.tsx                # 2D overlay: pitch ladder, compass tape, ALT/AGL/SPD/COORDS, sensor readouts
│   │   └── useDroneInput.ts            # keyboard state hook (reuses WASD/Space/Shift/QE key bindings)
│   ├── TerrainCanvas.tsx               # UNCHANGED
│   └── VoxelTerrain.tsx                # UNCHANGED
├── hooks/
│   └── useFlightMode.ts                # 'studio' | 'fpv' — lifted app-level state, drives sidebar collapse
└── App.tsx                             # mode toggle button, conditional sidebar render, spawn-point capture
```

This mirrors the existing pattern (`TerrainCanvas` / `VoxelTerrain` as siblings, `useInference.ts` as a dedicated hook) so it'll feel native to the codebase, not bolted on.

---

## 3. Mode Switching & Layout Behavior

- Add `useFlightMode()` — simple `useState<'studio' | 'fpv'>` lifted into `App.tsx`.
- **Entering FPV mode**:
  - Sidebar tabs (`01 SURFACE`, `02 FLOOD`, `03 PROFILE`, `04 INSPECT`) unmount or collapse to a thin edge strip — your call, but full unmount is simpler and better for the FPS budget (fewer components fighting for the frame).
  - Main viewport expands to fill the screen.
  - `DroneHUD` overlay mounts (absolute-positioned, plain HTML/Tailwind — no need for `drei`'s `<Html>` since this isn't anchored to a 3D object, it's a full-screen OSD like a real FPV goggle overlay).
  - The existing `Minimap.tsx` can optionally stay pinned in a corner as a tiny "radar" — free reuse, no new code, nice touch, purely optional.
- **Exiting FPV mode**: `Esc` key or a visible "Exit Drone Mode" button returns to Studio mode and restores the sidebar.
- **Pointer lock**: `TerrainCanvas` already uses pointer-lock-driven mouse look. Keep the two systems' pointer-lock state fully separate (don't share a ref) so entering/exiting FPV doesn't leave the Studio camera in a broken lock state.

### Spawn point selection
Simplest good option: while in Studio mode, add a "Set Drone Spawn" toggle. When active, a click on the terrain mesh raycasts to get world `(x, z)`, samples `dsm_raw` for ground height, and stores `spawnPoint = {x, y: terrainHeight + hoverAltitude, z}`. If the user never sets one, default to a fixed point centered above the tile at a safe altitude (e.g., `terrainMaxHeight + 15`). Either way, "on top of the map" as a default is fine — don't over-build this part.

---

## 4. Flight Physics (Reusing the Report's Own Blueprint)

The navigation audit you already have already worked out the math — implement it as-is inside `DroneFlightController.tsx`'s `useFrame`, rather than inventing a new model:

- **Velocity integration** (not direct position-set like the spectator cam):
  $$\vec{a} = \frac{\vec{F}_{\text{thrust}}}{m} - k_{\text{drag}}\vec{v}, \quad \vec{v}_{t+1} = \vec{v}_t + \vec{a}\Delta t, \quad \vec{p}_{t+1} = \vec{p}_t + \vec{v}_{t+1}\Delta t$$
- **Bank-into-turn roll** (automatic, not a separate control the player has to learn):
  $$\text{Roll}_{\text{target}} = -\text{clamp}(v_{\text{lateral}} \times 0.05 + \dot\psi \times 0.2,\ -0.45,\ 0.45)$$
  $$\text{Roll}_{t+1} = \text{lerp}(\text{Roll}_t, \text{Roll}_{\text{target}}, 10\Delta t)$$
- **Thrust-pitch coupling**: nose dips 5–15° under forward acceleration, levels out at hover.
- **Prop spin**: each rotor's angular speed scales with current throttle magnitude — purely cosmetic, driven off the same input state, done in `DroneModel.tsx`'s own `useFrame`.

### Recommended WASD mapping (keeps existing muscle memory, arcade-style rather than true RC-stick):
| Key | Effect |
|---|---|
| `W` / `S` | Pitch forward/back → forward/backward thrust |
| `A` / `D` | Yaw left/right (auto-banks per formula above) |
| `Space` / `Q` | Ascend (throttle up) |
| `E` | Descend (throttle down) |
| `Shift` | Turbo boost, same 3× multiplier as the existing spectator cam |

Default camera is **locked to the drone's nose** — no independent mouse-look. That's what makes it read as "FPV" rather than another free-fly camera. (A detachable/stabilized gimbal mode is a nice Phase-7 addition, not v1.)

### Why not a physics engine (Rapier/Cannon)?
The existing codebase has zero physics-engine dependency and a hard 60 FPS budget. A full rigid-body engine is overkill for one drone against a heightfield — it adds a non-trivial new dependency for a problem the project's own `dsm_raw` array already solves cheaply (see §5). Stick with the lightweight custom integrator above.

---

## 5. Terrain Collision & the Proximity Sensors (Left / Right / Bottom)

This is the part worth being precise about, since it's the most novel piece.

**Don't** use `THREE.Raycaster` against the actual terrain mesh for this. A 512×512 mesh is ~260k triangles, and mesh raycasting has no BVH acceleration by default in this project — it'd be the single most expensive thing in the frame loop. Instead, **reuse the exact `dsm_raw` array-sampling approach the navigation report already proposes** for AGL, and extend it to all three sensor directions:

```typescript
// Shared by all 3 sensors — steps outward from the drone in world space,
// sampling dsm_raw at each step, and reports the first distance where
// terrain height would intersect the drone's current altitude.
function castSensor(
  origin: THREE.Vector3,
  dirWorld: THREE.Vector3,   // normalized, already rotated by drone's quaternion
  dsmRaw: number[][],
  meshStats: MeshStats,
  maxRange = 50,
  step = 0.5
): number {
  for (let d = step; d <= maxRange; d += step) {
    const x = origin.x + dirWorld.x * d;
    const z = origin.z + dirWorld.z * d;
    const groundY = getTerrainElevationAt(x, z, dsmRaw, meshStats); // existing helper
    if (groundY >= origin.y - 0.2) return d; // obstacle/ground breaches drone altitude
  }
  return maxRange; // clear
}
```

- **Bottom sensor**: direction is straight down; this doubles as your AGL clearance value (`Altitude_drone >= Z_terrain(x,z) + 1.5m` from the report — use the same enforced-clearance rule to stop the drone clipping into rooftops).
- **Left/right sensors**: direction = drone's local `±X` axis, rotated into world space via the drone's current quaternion each frame, so they always point out the sides of the body regardless of heading.
- **Soft collision response**: if a sensor reading drops below a safety threshold (~1.5m), zero out the velocity component pointing into that direction rather than building a full physics bounce — a "soft bump," consistent with the arcade-physics approach above.

### Visualizing the 3 lines
Two complementary pieces, both driven by the same `castSensor` output:
1. **In-scene laser**: a `drei` `<Line>` (or a manually built `THREE.LineSegments`) from the sensor's arm anchor to the hit point, color-coded green (>10m) → yellow (3–10m) → red (<3m). Update its endpoint every frame from the ray result — cheap, it's just 2 points per line.
2. **HUD numeric readout**: `L: 12.4m` / `R: 8.2m` / `AGL: 3.1m` boxes in `DroneHUD.tsx`, same color coding, flashing on red.

This gives you both the literal "3 lines" visual you asked for *and* readable numbers, without adding a physics engine or expensive mesh raycasting.

---

## 6. HUD / OSD (`DroneHUD.tsx`)

Plain absolute-positioned overlay (Tailwind, matching the existing UI), not a 3D-anchored element:
- Center **pitch ladder / artificial horizon** (rotates with roll, translates with pitch).
- Top **compass tape** (rolling 000°–360° heading bar).
- Corner **telemetry boxes**: `ALT (MSL)`, `RADAR ALT (AGL)`, `SPD`, `COORDINATES` (project world XZ back through the GeoTIFF affine transform for real lat/lon or UTM, same transform `geospatial.py` already extracted).
- The 3 proximity readouts from §5.
- Center crosshair, small "DRONE MODE" badge, and an obvious Exit button.

---

## 7. Phased Build Order

| Phase | Deliverable |
|---|---|
| **0** | Mode toggle (`useFlightMode`), empty `components/drone/` scaffold, sidebar collapse wiring |
| **1** | Static `DroneModel.tsx` (body/arms/rotors/props/nav lights) placed at spawn point, nose-locked camera, empty HUD shell |
| **2** | Flight physics + WASD per §4, prop spin tied to throttle |
| **3** | AGL clearance / hard-floor & rooftop collision via `dsm_raw` |
| **4** | 3-sensor proximity array (§5): ray-march + laser lines + HUD readouts + soft-bump response |
| **5** | Full HUD/OSD polish (§6) |
| **6** | UX glue: Esc-to-exit, click-to-set spawn point, pointer-lock isolation from Studio mode |
| **7 (stretch)** | Autopilot orbit/transect flyby (already speced in the nav report §4.3), swap in a downloaded `.glb` model, selectable stabilized-gimbal camera mode |

Phases 0–5 are the "done, feels real" milestone. 6–7 are polish/stretch.

---

## 8. Dependencies

- **No new dependency required for v1.** Everything is vanilla Three.js math + existing R3F.
- **Recommended, not required**: `@react-three/drei` for `<Line>` and general convenience (`useGLTF` later if you go the model route). Check `frontend/package.json` first — if it's not already installed, `npm install @react-three/drei` inside `frontend/` (verify the installed version resolves cleanly against the pinned Three.js r174 before committing).
- Confirm before installing anything: this only touches the **frontend** `npm` workspace, not the backend `depthwizard` conda env — the project's "no ad-hoc environments" rule is a backend/Python rule and doesn't block a normal frontend npm install.

---

## 9. Guardrails (carried over from the project brief)

- Don't touch `geospatial.py`, `depth_estimator.py`, `calibration.py`, or the `dsm_raw` payload schema — this feature is 100% client-side, consuming `dsm_raw` as-is.
- Keep the 60 FPS budget: use `InstancedMesh` for the 4 rotors/props (repeated geometry), dispose geometries/textures on unmount, keep sensor ray-marching step count reasonable (0.5m steps over 50m max = 100 samples/sensor/frame — cheap, but don't lower the step size without reason).
- The existing spectator camera system (`TerrainCanvas.tsx`, `VoxelTerrain.tsx`) must not be modified or regressed by this work — it's a parallel system, not a refactor of the old one.

---


