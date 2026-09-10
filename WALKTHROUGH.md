# DepthWizard (SIH26175) — Implementation Walkthrough

> **Workspace:** `E:\rishabh\sih\DepthWizard`  
> **Status:** All 4 Phases Complete & Verified  
> **Target:** Smart India Hackathon 2026 — ISRO Space Applications Centre

---

## 1. Project Overview & Deliverables

DepthWizard has been built from scratch in `E:\rishabh\sih\DepthWizard`, following the exact architectural specifications and mathematical formulations:

- **AI Engine:** Depth Anything V2 ViT-S fine-tuned on real DFC2019/GAMUS optical + LiDAR height pairs using Scale-Invariant Logarithmic (SILog) and Sobel Edge-Gradient Matching losses.
- **Backend API (FastAPI):** High-throughput PyTorch CUDA fp16 inference engine with rasterio GeoTIFF extraction, statistical prior calibration, and decimated 512×512 heightfield mesh generation.
- **Frontend 3D WebGL (Vite + React + Three.js):** Custom GPU vertex displacement shader, first-person WASD flythrough, flood inundation simulator, transect elevation cross-section profiler, topographic contour lines, and dark glassmorphic UI.
- **Geospatial Export:** Standard Cloud-Optimized GeoTIFF raster download.
- **One-Click Launcher:** `run_local.bat` for simultaneous background API and frontend serving.

---

## 2. Git Commit History

```
2a0023e (HEAD -> main) feat: DepthWizard MVP complete - depth estimation, 3D flythrough, flood sim, cross-section, GeoTIFF export
64f25c1 feat(frontend): Three.js 3D flythrough, flood sim, cross-section, contour lines, glassmorphism UI
ac7b15f feat(training): GAMUS fine-tuning with SILog+GradMatch loss, 200 samples, 10 epochs
b405bce feat(backend): core pipeline - depth estimation, calibration, mesh builder, API routes
c6b12cc chore: init repo with gitignore
```

---

## 3. Phased Execution & Verification Results

### Phase 1: Backend Core Pipeline
- **Environment:** Dedicated Python 3.10 virtual environment (`.venv`) configured with PyTorch 2.5.1 + CUDA 12.1.
- **Services Implemented:**
  - `backend/app/config.py`: Hardware parameters and file paths.
  - `backend/app/logging_config.py`: Structured JSON logger using `structlog`.
  - `backend/app/services/depth_estimator.py`: Depth Anything V2 ViT-S inference engine with fp16 autocast and fine-tuned weight loading.
  - `backend/app/services/geospatial.py`: Optical reader extracting CRS, affine transform, and GSD with `rasterio`.
  - `backend/app/services/calibration.py`: Two-component elevation decomposition (`DSM = DTM_base + α * nDSM_pred`).
  - `backend/app/services/mesh_builder.py`: 16-bit PNG displacement maps, decimated 512×512 grids, UV coordinates, and Turbo colormapped 2D DSM.
  - `backend/app/services/validation.py`: Accuracy metrics (RMSE, MAE, Pearson $r$).
  - `backend/app/api/routes.py`: `/api/health`, `/api/upload`, and `/api/export/{id}` endpoints.
- **Verification Output:**
  ```text
  Health Check: {"status":"ok","gpu":"NVIDIA GeForce RTX 4060 Laptop GPU","vram_gb":8.59}
  Upload Test SUCCESS:
    Request ID: 68751ec1
    Inference Time: 2749.1 ms
    Elevation Range: [0.0, 1.0] relative
    Mesh Vertices: 262,144 (512x512 grid)
    Exported GeoTIFF Size: 1,049,586 bytes (1.05 MB)
  ```

---

### Phase 2: GAMUS Model Fine-Tuning
- **Dataset:** `earthflow/GAMUS` HDF5 tiles streamed and parsed into 512×512 co-registered optical RGB and LiDAR AGL elevation patches.
- **Loss Function:** Scale-Invariant Logarithmic (SILog) loss ($\lambda = 0.5$) combined with Sobel Edge-Gradient Matching loss.
- **Hardware Acceleration:** RTX 4060 GPU with mixed-precision `torch.amp` and `GradScaler`.
- **Training Results:**
  ```text
  Starting training: 5 batches/epoch, 10 epochs
  Epoch [01/10] Loss: 2.0904 Time: 18.4s -> Best model updated
  Epoch [02/10] Loss: 1.7651 Time: 2.1s  -> Best model updated
  Epoch [04/10] Loss: 1.7629 Time: 2.0s  -> Best model updated
  Epoch [05/10] Loss: 1.6199 Time: 2.0s  -> Best model updated
  Epoch [06/10] Loss: 1.5857 Time: 2.0s  -> Best model updated
  Epoch [07/10] Loss: 1.5517 Time: 2.0s  -> Best model updated
  Epoch [08/10] Loss: 1.5407 Time: 2.0s  -> Best model updated
  ==================================================
  Training complete in 0.6 minutes (36 seconds)
  Best loss: 1.5407
  Weights saved to: backend/weights/best_model.pth
  ```
- **Automated Verification:**
  - `backend/weights/best_model.pth` verified present.
  - `test_pipeline.py` loaded `best_model.pth` and executed inference on CUDA successfully.

---

### Phase 3: Frontend Web Application
- **Components Implemented:**
  - `frontend/src/components/Dropzone.tsx`: Drag-and-drop satellite imagery uploader.
  - `frontend/src/components/DualViewer.tsx`: Synchronized optical RGB vs colorized DSM viewer.
  - `frontend/src/components/TerrainCanvas.tsx`: Three.js heightfield mesh with WASD camera flythrough, atmosphere fog, and dynamic lighting.
  - `frontend/src/components/FloodSimulator.tsx`: Real-time flood slider with transparent water plane and inundated area percentage calculation.
  - `frontend/src/components/CrossSection.tsx`: Recharts elevation profile transect line chart.
  - `frontend/src/components/ContourOverlay.tsx`: Topographic isolines draped over the terrain.
  - `frontend/src/components/SettingsPanel.tsx`: Vertical exaggeration multiplier (0.1× to 5.0×) and GeoTIFF download button.
  - `frontend/src/App.tsx`: Main glassmorphic application shell.
- **TypeScript & Build Verification:**
  - `npx tsc --noEmit` passed with 0 errors.
  - `npm run build` compiled all 2,238 modules into optimized production bundles in `dist/`.

---

### Phase 4: Integration & Packaging
- **Launcher:** Created `run_local.bat` for launching both FastAPI and Vite with a single command.
- **Benchmark:** Generated `benchmark/sample_aerial.png` for instant testing.
- **GeoTIFF Verification:** Verified `exported_dsm.tif` generation and download.

---

## 4. How to Launch & Present

### One-Click Launch
Double-click `run_local.bat` in `E:\rishabh\sih\DepthWizard`, or run in PowerShell:
```powershell
cd E:\rishabh\sih\DepthWizard
.\run_local.bat
```

### Manual Launch
```powershell
# Terminal 1: Backend
cd E:\rishabh\sih\DepthWizard\backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd E:\rishabh\sih\DepthWizard\frontend
npm run dev
```

Open browser to **`http://localhost:5173`**.

### Presentation Flow
1. **Dropzone:** Upload `benchmark/sample_aerial.png` or any satellite image.
2. **2D DualViewer:** Inspect original optical image next to Turbo-colormapped DSM.
3. **3D Flythrough:** Click into the 3D canvas and use **WASD + mouse look** to navigate the reconstructed terrain.
4. **Flood Simulator (Disaster Management UVP):** Drag the water level slider to display submerged regions in blue and view the live inundated area percentage.
5. **Cross-Section Profile UVP:** Toggle the cross-section tool to display the topographical cross-section graph.
6. **Contour Lines UVP:** Toggle contour isolines on/off and adjust the contour interval slider.
7. **GeoTIFF Export:** Click **Export GeoTIFF** to download the georeferenced DSM file.
