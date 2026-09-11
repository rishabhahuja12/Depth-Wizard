# App-Integration Log

Most upgrade work lives in `upgrade/` and is deleted by removing that folder.
A few changes **must** edit the real app (`frontend/`, `backend/`) to take effect —
those are NOT reverted by deleting `upgrade/`. Every such edit is logged here with
its rationale and exact revert, so the "one-delete" story stays honest and the app
can always be returned to baseline.

> To undo everything the upgrade touched: delete `upgrade/`, then `git revert` each
> commit listed below (or `git checkout <baseline>` the named files).

---

## P3a — Triplanar terrain texturing (mesh "melted-tent" fix)

- **Date:** 2026-09-11
- **File(s):** `frontend/src/components/TerrainCanvas.tsx`
- **Why:** a single top-down texture UV-draped over the smooth mesh stretches down
  the steep diagonal faces (root cause #3 of the "melted-tent" artifact). Triplanar
  sampling blends three axis-projected samples by the true surface steepness, so
  vertical faces stop smearing.
- **What changed (additive, surgical):**
  - Added a `uTriYScale` uniform (side-projection vertical tiling rate).
  - Replaced the `#include <map_fragment>` chunk in the existing `onBeforeCompile`
    with a triplanar sampler. The true normal is recovered from screen-space
    derivatives of the displaced world position (`dFdx/dFdy`), so it works despite
    GPU displacement. Flat/up-facing areas keep the exact original UV sampling
    (blend.y ≈ 1) — only steep faces change.
  - Renamed `customProgramCacheKey` → `terrain_triplanar_contour_mat` so the new
    shader variant doesn't collide with a cached program.
- **Not included (deliberately):** edge-aware depth *sharpening* (the other half of
  P3a) — best driven by the P1 metric model's sharper edges or a heightmap filter;
  deferred so this stays a low-risk, top-preserving change.
- **Verification:** `npm run build` passes (tsc + vite). Visual confirmation of the
  in-browser shader is pending a live run (smooth-mesh render mode).
- **Revert:** `git checkout <pre-P3a commit> -- frontend/src/components/TerrainCanvas.tsx`
  (or `git revert` the P3a commit). No other files affected.
