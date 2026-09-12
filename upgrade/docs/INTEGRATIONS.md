# App-Integration Log

Most upgrade work lives in `upgrade/` and is deleted by removing that folder.
A few changes **must** edit the real app (`frontend/`, `backend/`) to take effect —
those are NOT reverted by deleting `upgrade/`. Every such edit is logged here with
its rationale and exact revert, so the "one-delete" story stays honest and the app
can always be returned to baseline.

> To undo everything the upgrade touched: delete `upgrade/`, then `git revert` each
> commit listed below (or `git checkout <baseline>` the named files).

---

## P3a — Triplanar terrain texturing — ❌ TRIED & REVERTED

- **Dates:** added 2026-09-11 (`d07730e`), reverted same day.
- **File(s):** `frontend/src/components/TerrainCanvas.tsx` (now back to baseline).
- **What was tried:** triplanar map sampling in `onBeforeCompile` (side-projected
  the texture on steep faces, blended by a `dFdx/dFdy` surface normal, `uTriYScale`
  uniform) to stop the top-down texture smearing down vertical faces.
- **Why reverted (verified in a live render):**
  1. **Wrong lever.** The "melted-tent" look is a **geometry/topology** problem —
     a continuous displaced `PlaneGeometry` cannot form vertical walls, so buildings
     are smooth ramps/blobs. Texturing cannot change that silhouette.
  2. **It added a new artifact.** A satellite ortho texture does NOT tile, but the
     triplanar side-projection sampled it with `fract(... worldY ...)`, repeating the
     whole image down each face → visible **horizontal banding**. Confirmed on-screen
     by the user (running commit `d07730e`).
- **Lesson for P3:** the melted-tent fix must be **geometric** — vertical skirt
  quads at depth discontinuities (P3b) or LOD1 extrusion — OR lean on **voxel mode**,
  which already renders correct vertical faces (and is the default). Triplanar only
  ever mitigates texture smear on faces that already exist; it can't create walls.
- **Status:** `TerrainCanvas.tsx` restored to its pre-P3a baseline. App unchanged.
