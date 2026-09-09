# DepthWizard 🧙‍♂️

**Single-View Height Estimation & 3D Flythrough**
*Smart India Hackathon 2026 — Problem Statement SIH26175 (ISRO SAC)*

DepthWizard is an agile AI-powered geospatial platform that reconstructs accurate Digital Surface Models (DSMs) from a single overhead optical RGB satellite image and renders an interactive, low-latency 3D WebGL flythrough environment.

---

## 🌟 Unique Value Propositions (UVPs)

1. **🌊 Real-Time Flood Inundation Simulator:** Interactive water-level threshold slider with instantaneous GPU area submergence calculation.
2. **✂️ Elevation Cross-Section Profiler:** Real-time transect sampling and 2D topography curve rendering.
3. **🗺️ Topographic Contour Isolines:** Dynamic `d3-contour` elevation lines draped directly over 3D terrain geometry.
4. **📥 Standard OGC GeoTIFF Export:** Cloud-Optimized GeoTIFF generation with embedded projection & affine coordinates.
5. **📐 Academically Grounded SILog Loss:** Supervised Scale-Invariant Logarithmic & Edge-Gradient loss based on Depth Anything V2 & Sat3R (CVPR 2026).

---

## 🚀 Quick Start

### 1. Requirements
- Windows 11 / Linux
- NVIDIA GPU with CUDA support (e.g. RTX 4060)
- Python 3.10+
- Node.js 18+

### 2. Automated Run
Simply double-click:
```bat
run_local.bat
```
Or run manually:
```bash
# Terminal 1: Backend
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Terminal 2: Frontend
cd frontend
npm run dev
```

Open your browser to: **`http://localhost:5173`**

---

## 🏗️ Architecture & Stack
- **AI Backbone:** Depth Anything V2 ViT-S (`depth-anything/Depth-Anything-V2-Small-hf`)
- **Fine-Tuning Dataset:** `earthflow/GAMUS` (DFC2019 Jacksonville/Omaha co-registered optical + nDSM LiDAR pairs)
- **Backend API:** FastAPI, PyTorch (CUDA fp16 mixed precision), Rasterio, Scipy, Structlog
- **Frontend Engine:** Vite, React 18, Three.js, React Three Fiber, Tailwind CSS v4, Recharts
