# Graph Report - DepthWizard  (2026-09-10)

## Corpus Check
- Corpus is ~7,911 words - fits in a single context window. You may not need a graph.

## Summary
- 220 nodes · 311 edges · 18 communities (11 shown, 3 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 5 edges (avg confidence: 0.92)
- Token cost: 1,200 input · 850 output

## Community Hubs (Navigation)
- Frontend React UI & Flythrough Shell
- GAMUS Training & Fine-Tuning Pipeline
- Frontend Project Configuration
- FastAPI Routes & GeoTIFF Export
- TypeScript Compiler Options
- Heightfield Mesh Builder & GPU Displacement
- Build Tools & Platform Binaries
- Geospatial Reader & Metadata Parser
- Two-Component Elevation Calibration
- UI Libraries & Geospatial Dependencies
- Depth Anything V2 Inference Engine
- CUDA Hardware Verification Suite
- Synthetic Aerial Benchmark Suite
- Turbo Colormap Palette Shaders

## God Nodes (most connected - your core abstractions)
1. `compilerOptions` - 16 edges
2. `react` - 12 edges
3. `calibrate_depth()` - 10 edges
4. `DepthEstimator` - 9 edges
5. `build_mesh_data()` - 8 edges
6. `DepthWizard System Overview` - 8 edges
7. `upload_image()` - 7 edges
8. `read_image()` - 7 edges
9. `GAMUSDataset` - 7 edges
10. `lucide-react` - 7 edges

## Surprising Connections (you probably didn't know these)
- `SIH26175 Verification Walkthrough` --references--> `DepthWizard System Overview`  [EXTRACTED]
  WALKTHROUGH.md → README.md
- `upload_image()` --uses--> `InferenceResponse`  [INFERRED]
  backend/app/api/routes.py → backend/app/api/schemas.py
- `health()` --uses--> `HealthResponse`  [INFERRED]
  backend/app/api/routes.py → backend/app/api/schemas.py
- `upload_image()` --calls--> `calibrate_depth()`  [EXTRACTED]
  backend/app/api/routes.py → backend/app/services/calibration.py
- `upload_image()` --calls--> `build_mesh_data()`  [EXTRACTED]
  backend/app/api/routes.py → backend/app/services/mesh_builder.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Disaster Management & Topography UVP Suite** — uvp_flood_simulator, uvp_cross_section, uvp_contour_lines, uvp_geotiff_export [INFERRED 0.95]

## Communities (18 total, 3 thin omitted)

### Community 0 - "Frontend React UI & Flythrough Shell"
Cohesion: 0.08
Nodes (30): App(), ColorBar(), ColorBarProps, ContourOverlay(), ContourOverlayProps, CrossSection(), CrossSectionProps, ProfilePoint (+22 more)

### Community 1 - "GAMUS Training & Fine-Tuning Pipeline"
Cohesion: 0.10
Nodes (16): GAMUSDataset, PyTorch Dataset wrapping HuggingFace earthflow/GAMUS HDF5 files., CombinedLoss, GradientMatchingLoss, Scale-Invariant Logarithmic Loss (SILog) + Gradient Matching Loss. References:…, SILogLoss, DepthWizard MVP Training Script. Fine-tunes Depth Anything V2 ViT-S on GAMUS…, train() (+8 more)

### Community 2 - "Frontend Project Configuration"
Cohesion: 0.08
Nodes (25): name, private, scripts, build, dev, preview, type, version (+17 more)

### Community 3 - "FastAPI Routes & GeoTIFF Export"
Cohesion: 0.17
Nodes (15): export_dsm(), health(), Download computed DSM as GeoTIFF., set_estimator(), ErrorResponse, HealthResponse, InferenceResponse, setup_logging() (+7 more)

### Community 4 - "TypeScript Compiler Options"
Cohesion: 0.11
Nodes (17): compilerOptions, allowImportingTsExtensions, isolatedModules, jsx, lib, module, moduleResolution, noEmit (+9 more)

### Community 5 - "Heightfield Mesh Builder & GPU Displacement"
Cohesion: 0.18
Nodes (13): apply_turbo_colormap(), build_mesh_data(), ndarray, Heightfield mesh generation with mathematically correct vertices, faces, UVs,…, Build heightfield mesh data from DSM. Returns dict with: - heightmap_b64:…, Apply Turbo colormap to normalized [0,1] values. Returns (H, W, 3) uint8 RGB…, compute_metrics(), ndarray (+5 more)

### Community 6 - "Build Tools & Platform Binaries"
Cohesion: 0.14
Nodes (14): devDependencies, lightningcss-win32-x64-msvc, @rollup/rollup-win32-x64-msvc, tailwindcss, @tailwindcss/oxide-win32-x64-msvc, @tailwindcss/vite, @types/d3-contour, @types/d3-geo (+6 more)

### Community 7 - "Geospatial Reader & Metadata Parser"
Cohesion: 0.24
Nodes (11): upload_image(), ImageMetadata, GeoTIFF and standard image reader. Detects CRS, affine transform, and…, Read an image from bytes. Returns (rgb_array, pil_image, metadata). Supports…, Read GeoTIFF with rasterio, extract CRS and transform., Read PNG/JPG with PIL., _read_geotiff(), read_image() (+3 more)

### Community 8 - "Two-Component Elevation Calibration"
Cohesion: 0.36
Nodes (9): calibrate_depth(), CalibrationResult, _metric_calibration(), ndarray, Two-Component Elevation Decomposition. DSM(x,y) = DTM_base(x,y) + α ·…, Convert relative depth [0,1] to calibrated DSM. For georeferenced: scale to…, Statistical prior calibration. Assumes urban scene with buildings up to…, Keep as relative depth normalized to [0, 1]. (+1 more)

### Community 9 - "UI Libraries & Geospatial Dependencies"
Cohesion: 0.20
Nodes (10): dependencies, d3-contour, d3-geo, lucide-react, react, react-dom, @react-three/drei, @react-three/fiber (+2 more)

### Community 10 - "Depth Anything V2 Inference Engine"
Cohesion: 0.33
Nodes (5): ndarray, Predict relative depth from an RGB PIL Image. Args: rgb_image: PIL Image in RGB…, MC Dropout inference for uncertainty estimation. Returns (mean_depth,…, Image, no_grad

## Knowledge Gaps
- **74 isolated node(s):** `name`, `private`, `version`, `type`, `dev` (+69 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 115 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DepthWizard System Overview` connect `GAMUS Training & Fine-Tuning Pipeline` to `Frontend React UI & Flythrough Shell`, `FastAPI Routes & GeoTIFF Export`?**
  _High betweenness centrality (0.385) - this node is a cross-community bridge._
- **Why does `UVP: Flood Inundation Simulator` connect `Frontend React UI & Flythrough Shell` to `GAMUS Training & Fine-Tuning Pipeline`?**
  _High betweenness centrality (0.187) - this node is a cross-community bridge._
- **Why does `react` connect `Frontend React UI & Flythrough Shell` to `Frontend Project Configuration`?**
  _High betweenness centrality (0.164) - this node is a cross-community bridge._
- **What connects `name`, `private`, `version` to the rest of the system?**
  _74 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Frontend React UI & Flythrough Shell` be split into smaller, more focused modules?**
  _Cohesion score 0.07641196013289037 - nodes in this community are weakly interconnected._
- **Should `GAMUS Training & Fine-Tuning Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.09788359788359788 - nodes in this community are weakly interconnected._
- **Should `Frontend Project Configuration` be split into smaller, more focused modules?**
  _Cohesion score 0.07977207977207977 - nodes in this community are weakly interconnected._