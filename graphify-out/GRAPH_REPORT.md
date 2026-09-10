# Graph Report - .  (2026-09-10)

## Corpus Check
- 46 files · ~1,988,331 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 255 nodes · 211 edges · 92 communities detected
- Extraction: 92% EXTRACTED · 8% INFERRED · 0% AMBIGUOUS · INFERRED: 17 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 86|Community 86]]
- [[_COMMUNITY_Community 87|Community 87]]
- [[_COMMUNITY_Community 88|Community 88]]
- [[_COMMUNITY_Community 89|Community 89]]
- [[_COMMUNITY_Community 90|Community 90]]
- [[_COMMUNITY_Community 91|Community 91]]

## God Nodes (most connected - your core abstractions)
1. `CombinedLoss` - 14 edges
2. `FullGAMUSDataset` - 8 edges
3. `PrecisionGAMUSDataset` - 8 edges
4. `TeeLogger` - 7 edges
5. `DepthWizard System Overview` - 7 edges
6. `GAMUSDataset` - 6 edges
7. `TeeLogger` - 6 edges
8. `build_presentation()` - 5 edges
9. `read_image()` - 5 edges
10. `StableSILogLoss` - 5 edges

## Surprising Connections (you probably didn't know these)
- `DepthWizard: Precision Fine-Tuning Engine (Stage 2 Refinement). Bootstraps from` --uses--> `CombinedLoss`  [INFERRED]
  backend\training\train_precision_refinement.py → backend\training\losses.py
- `Loads tiles from local cache and extracts non-degenerate 512x512 patches.` --uses--> `CombinedLoss`  [INFERRED]
  backend\training\train_precision_refinement.py → backend\training\losses.py
- `DepthWizard Training Script — v2 (bug-fixed + stronger). Fine-tunes Depth Anythi` --uses--> `CombinedLoss`  [INFERRED]
  backend\training\train_gamus.py → backend\training\losses.py
- `DepthWizard: Full Dataset Overnight Training Engine (100% earthflow/GAMUS).  Spe` --uses--> `CombinedLoss`  [INFERRED]
  backend\training\train_gamus_full.py → backend\training\losses.py
- `Simultaneously writes console output to terminal and log file.` --uses--> `CombinedLoss`  [INFERRED]
  backend\training\train_gamus_full.py → backend\training\losses.py

## Hyperedges (group relationships)
- **Disaster Management & Topography UVP Suite** — uvp_flood_simulator, uvp_cross_section, uvp_contour_lines, uvp_geotiff_export [INFERRED 0.95]

## Communities

### Community 0 - "Community 0"
Cohesion: 0.18
Nodes (9): CombinedLoss, download_tile_pair(), FullGAMUSDataset, DepthWizard: Full Dataset Overnight Training Engine (100% earthflow/GAMUS).  Spe, Simultaneously writes console output to terminal and log file., Downloads an RGB and AGL pair with automatic retry on transient drops., Lazy HDF5 Disk Streaming Dataset for 100% GAMUS (15,348 patches)., run_training() (+1 more)

### Community 1 - "Community 1"
Cohesion: 0.12
Nodes (3): DepthEstimator, Depth Anything V2 ViT-S inference engine. Uses HuggingFace transformers for load, Test core pipeline services: depth estimation, calibration, and mesh building.

### Community 2 - "Community 2"
Cohesion: 0.22
Nodes (8): BaseModel, export_dsm(), _process_pipeline(), Download computed DSM as GeoTIFF without Windows file lock race., Synchronous pipeline executed in threadpool to prevent event-loop blocking., ErrorResponse, HealthResponse, InferenceResponse

### Community 3 - "Community 3"
Cohesion: 0.17
Nodes (11): create_slide2_flowchart(), create_slide3_architecture(), create_slide4_swot_risks(), create_slide5_impact_ecosystem(), create_slide6_research_lineage(), Generate High-Resolution Diagrammatic Assets for DepthWizard SIH 2026 Presentati, Slide 4: Feasibility SWOT Matrix + Risk-Mitigation Chevrons., Slide 5: 4-Quadrant Impact Ecosystem Hub & Key Metrics. (+3 more)

### Community 4 - "Community 4"
Cohesion: 0.23
Nodes (5): PrecisionGAMUSDataset, DepthWizard: Precision Fine-Tuning Engine (Stage 2 Refinement). Bootstraps from, Loads tiles from local cache and extracts non-degenerate 512x512 patches., run_precision_training(), TeeLogger

### Community 5 - "Community 5"
Cohesion: 0.25
Nodes (10): add_card(), build_presentation(), find_shape_by_text(), format_bullet(), Build Official SIH 2026 Presentation for DepthWizard (SIH26175 - ISRO SAC). Adhe, Safely removes a shape from a slide., Finds a shape containing specific text., Adds a styled rounded rectangular card container. (+2 more)

### Community 6 - "Community 6"
Cohesion: 0.18
Nodes (0): 

### Community 7 - "Community 7"
Cohesion: 0.2
Nodes (4): Dataset, GAMUSDataset, PyTorch Dataset wrapping HuggingFace earthflow/GAMUS HDF5 files.  Fixes applied:, DepthWizard Training Script — v2 (bug-fixed + stronger). Fine-tunes Depth Anythi

### Community 8 - "Community 8"
Cohesion: 0.27
Nodes (4): GradientMatchingLoss, Mathematically Stabilized Precision Loss for DepthWizard. Combines: 1. StableSIL, Log1p Scale-Invariant Logarithmic Loss.     d = ln(1 + gamma * p) - ln(1 + gamma, StableSILogLoss

### Community 9 - "Community 9"
Cohesion: 0.33
Nodes (8): calibrate_depth(), CalibrationResult, _metric_calibration(), Two-Component Elevation Decomposition. DSM(x,y) = DTM_base(x,y) + α · nDSM_pred(, Keep as relative depth normalized to [0, 1]., Convert relative depth [0,1] to calibrated DSM.      For georeferenced: scale to, Two-Component Elevation Decomposition:     DSM(x, y) = DTM_base(x, y) + α · nDSM, _relative_calibration()

### Community 10 - "Community 10"
Cohesion: 0.31
Nodes (8): ImageMetadata, GeoTIFF and standard image reader. Detects CRS, affine transform, and geospatial, Read PNG/JPG with PIL., Read an image from bytes. Returns (rgb_array, pil_image, metadata).     Supports, Read GeoTIFF with rasterio, extract CRS and transform., _read_geotiff(), read_image(), _read_standard()

### Community 11 - "Community 11"
Cohesion: 0.32
Nodes (7): apply_turbo_colormap(), build_mesh_data(), compute_normal_map(), Heightfield mesh generation with mathematically correct vertices, faces, UVs, no, Apply Turbo colormap to normalized [0,1] values.     Returns (H, W, 3) uint8 RGB, Compute tangent-space normal map from DSM elevation grid.     Returns (H, W, 3), Build heightfield mesh data from DSM.      Returns dict with:       - heightmap_

### Community 12 - "Community 12"
Cohesion: 0.25
Nodes (8): earthflow/GAMUS Dataset (DFC2019), Scale-Invariant Logarithmic (SILog) Loss, DepthWizard System Overview, SIH26175 Verification Walkthrough, UVP: Topographic Contour Isolines, UVP: Elevation Cross-Section Profiler, UVP: Flood Inundation Simulator, UVP: OGC Cloud-Optimized GeoTIFF Export

### Community 13 - "Community 13"
Cohesion: 0.38
Nodes (6): compute_metrics(), ensure_val_tiles(), evaluate_benchmark(), DepthWizard: Official GAMUS Validation Benchmark Script. Evaluates Base Pretrain, Downloads official val tiles from earthflow/GAMUS if not already on disk., Computes academic monocular depth estimation metrics on normalized elevation.

### Community 14 - "Community 14"
Cohesion: 0.5
Nodes (4): compute_metrics(), Compute RMSE, MAE, Pearson r between predicted and reference DSMs., Compute accuracy metrics between predicted and reference DSMs.     Both must be, ValidationMetrics

### Community 15 - "Community 15"
Cohesion: 0.5
Nodes (2): handleInputChange(), validateFile()

### Community 16 - "Community 16"
Cohesion: 0.67
Nodes (0): 

### Community 17 - "Community 17"
Cohesion: 0.67
Nodes (1): Test CUDA device and PyTorch installation.

### Community 18 - "Community 18"
Cohesion: 0.67
Nodes (1): Generate a sample aerial RGB image for testing.

### Community 19 - "Community 19"
Cohesion: 0.67
Nodes (0): 

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (2): generateColormapLUT(), turboColormap()

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (0): 

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (1): DepthWizard: Production Server Launcher. Starts FastAPI backend on http://127.0.

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (0): 

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (0): 

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (0): 

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (0): 

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (0): 

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (0): 

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (0): 

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (0): 

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (0): 

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Predict relative depth from an RGB PIL Image.          Args:             rgb_ima

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): MC Dropout inference for uncertainty estimation.         Returns (mean_depth, co

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (0): 

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (0): 

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (0): 

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (0): 

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (0): 

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (0): 

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (0): 

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (0): 

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (0): 

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (0): 

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (0): 

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (0): 

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): lucide-react

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): react

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): @react-three/fiber

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): recharts

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): three

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): PyTorch Dataset wrapping HuggingFace earthflow/GAMUS HDF5 files.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Scale-Invariant Logarithmic Loss (SILog) + Gradient Matching Loss. References:…

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): DepthWizard MVP Training Script. Fine-tunes Depth Anything V2 ViT-S on GAMUS…

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): Predict relative depth from an RGB PIL Image. Args: rgb_image: PIL Image in RGB…

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): MC Dropout inference for uncertainty estimation. Returns (mean_depth,…

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): Test CUDA device and PyTorch installation.

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): Generate a sample aerial RGB image for testing.

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): d3-contour

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): d3-geo

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): lightningcss-win32-x64-msvc

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): react-dom

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): @react-three/drei

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): @rollup/rollup-win32-x64-msvc

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): tailwindcss

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): @tailwindcss/oxide-win32-x64-msvc

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): @tailwindcss/vite

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): @types/d3-contour

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (1): @types/d3-geo

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (1): @types/react

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (1): @types/react-dom

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (1): @types/three

### Community 72 - "Community 72"
Cohesion: 1.0
Nodes (1): typescript

### Community 73 - "Community 73"
Cohesion: 1.0
Nodes (1): vite

### Community 74 - "Community 74"
Cohesion: 1.0
Nodes (1): @vitejs/plugin-react

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (1): Download computed DSM as GeoTIFF.

### Community 76 - "Community 76"
Cohesion: 1.0
Nodes (1): Depth Anything V2 ViT-S inference engine. Uses HuggingFace transformers for…

### Community 77 - "Community 77"
Cohesion: 1.0
Nodes (1): Heightfield mesh generation with mathematically correct vertices, faces, UVs,…

### Community 78 - "Community 78"
Cohesion: 1.0
Nodes (1): Build heightfield mesh data from DSM. Returns dict with: - heightmap_b64:…

### Community 79 - "Community 79"
Cohesion: 1.0
Nodes (1): Apply Turbo colormap to normalized [0,1] values. Returns (H, W, 3) uint8 RGB…

### Community 80 - "Community 80"
Cohesion: 1.0
Nodes (1): Compute RMSE, MAE, Pearson r between predicted and reference DSMs.

### Community 81 - "Community 81"
Cohesion: 1.0
Nodes (1): Compute accuracy metrics between predicted and reference DSMs. Both must be (H,…

### Community 82 - "Community 82"
Cohesion: 1.0
Nodes (1): Test core pipeline services: depth estimation, calibration, and mesh building.

### Community 83 - "Community 83"
Cohesion: 1.0
Nodes (1): GeoTIFF and standard image reader. Detects CRS, affine transform, and…

### Community 84 - "Community 84"
Cohesion: 1.0
Nodes (1): Read an image from bytes. Returns (rgb_array, pil_image, metadata). Supports…

### Community 85 - "Community 85"
Cohesion: 1.0
Nodes (1): Read GeoTIFF with rasterio, extract CRS and transform.

### Community 86 - "Community 86"
Cohesion: 1.0
Nodes (1): Read PNG/JPG with PIL.

### Community 87 - "Community 87"
Cohesion: 1.0
Nodes (1): Two-Component Elevation Decomposition. DSM(x,y) = DTM_base(x,y) + α ·…

### Community 88 - "Community 88"
Cohesion: 1.0
Nodes (1): Convert relative depth [0,1] to calibrated DSM. For georeferenced: scale to…

### Community 89 - "Community 89"
Cohesion: 1.0
Nodes (1): Statistical prior calibration. Assumes urban scene with buildings up to…

### Community 90 - "Community 90"
Cohesion: 1.0
Nodes (1): Keep as relative depth normalized to [0, 1].

### Community 91 - "Community 91"
Cohesion: 1.0
Nodes (1): Sample Aerial Overhead Imagery

## Knowledge Gaps
- **90 isolated node(s):** `Build Official SIH 2026 Presentation for DepthWizard (SIH26175 - ISRO SAC). Adhe`, `Safely removes a shape from a slide.`, `Finds a shape containing specific text.`, `Adds a styled rounded rectangular card container.`, `Formats a bullet point with a bold colored prefix followed by normal text.` (+85 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 21`** (2 nodes): `test_browser_samples.py`, `run_browser_tests()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (2 nodes): `run_server.py`, `DepthWizard: Production Server Launcher. Starts FastAPI backend on http://127.0.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (2 nodes): `logging_config.py`, `setup_logging()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (2 nodes): `ColorBar()`, `ColorBar.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (2 nodes): `setDelta()`, `FloodSimulator.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (2 nodes): `useInference.ts`, `useInference()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `inspect_template.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `verify_presentation.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `config.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Predict relative depth from an RGB PIL Image.          Args:             rgb_ima`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `MC Dropout inference for uncertainty estimation.         Returns (mean_depth, co`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `vite.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `main.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `ContourOverlay.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `CrossSection.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `DualViewer.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `SettingsPanel.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `merge_extract.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `run_ast.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `run_build.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `run_label.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `lucide-react`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `react`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `@react-three/fiber`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `recharts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `three`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `PyTorch Dataset wrapping HuggingFace earthflow/GAMUS HDF5 files.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Scale-Invariant Logarithmic Loss (SILog) + Gradient Matching Loss. References:…`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `DepthWizard MVP Training Script. Fine-tunes Depth Anything V2 ViT-S on GAMUS…`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `Predict relative depth from an RGB PIL Image. Args: rgb_image: PIL Image in RGB…`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `MC Dropout inference for uncertainty estimation. Returns (mean_depth,…`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `Test CUDA device and PyTorch installation.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `Generate a sample aerial RGB image for testing.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `d3-contour`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `d3-geo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `lightningcss-win32-x64-msvc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `react-dom`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `@react-three/drei`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `@rollup/rollup-win32-x64-msvc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `tailwindcss`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `@tailwindcss/oxide-win32-x64-msvc`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `@tailwindcss/vite`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `@types/d3-contour`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `@types/d3-geo`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `@types/react`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (1 nodes): `@types/react-dom`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (1 nodes): `@types/three`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 72`** (1 nodes): `typescript`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 73`** (1 nodes): `vite`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 74`** (1 nodes): `@vitejs/plugin-react`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 75`** (1 nodes): `Download computed DSM as GeoTIFF.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 76`** (1 nodes): `Depth Anything V2 ViT-S inference engine. Uses HuggingFace transformers for…`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 77`** (1 nodes): `Heightfield mesh generation with mathematically correct vertices, faces, UVs,…`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 78`** (1 nodes): `Build heightfield mesh data from DSM. Returns dict with: - heightmap_b64:…`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 79`** (1 nodes): `Apply Turbo colormap to normalized [0,1] values. Returns (H, W, 3) uint8 RGB…`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 80`** (1 nodes): `Compute RMSE, MAE, Pearson r between predicted and reference DSMs.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 81`** (1 nodes): `Compute accuracy metrics between predicted and reference DSMs. Both must be (H,…`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 82`** (1 nodes): `Test core pipeline services: depth estimation, calibration, and mesh building.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 83`** (1 nodes): `GeoTIFF and standard image reader. Detects CRS, affine transform, and…`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 84`** (1 nodes): `Read an image from bytes. Returns (rgb_array, pil_image, metadata). Supports…`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 85`** (1 nodes): `Read GeoTIFF with rasterio, extract CRS and transform.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 86`** (1 nodes): `Read PNG/JPG with PIL.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 87`** (1 nodes): `Two-Component Elevation Decomposition. DSM(x,y) = DTM_base(x,y) + α ·…`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 88`** (1 nodes): `Convert relative depth [0,1] to calibrated DSM. For georeferenced: scale to…`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 89`** (1 nodes): `Statistical prior calibration. Assumes urban scene with buildings up to…`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 90`** (1 nodes): `Keep as relative depth normalized to [0, 1].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 91`** (1 nodes): `Sample Aerial Overhead Imagery`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `CombinedLoss` connect `Community 0` to `Community 8`, `Community 4`, `Community 7`?**
  _High betweenness centrality (0.025) - this node is a cross-community bridge._
- **Why does `DepthWizard Training Script — v2 (bug-fixed + stronger). Fine-tunes Depth Anythi` connect `Community 7` to `Community 0`?**
  _High betweenness centrality (0.007) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `CombinedLoss` (e.g. with `DepthWizard Training Script — v2 (bug-fixed + stronger). Fine-tunes Depth Anythi` and `TeeLogger`) actually correct?**
  _`CombinedLoss` has 11 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Build Official SIH 2026 Presentation for DepthWizard (SIH26175 - ISRO SAC). Adhe`, `Safely removes a shape from a slide.`, `Finds a shape containing specific text.` to the rest of the system?**
  _90 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.12 - nodes in this community are weakly interconnected._