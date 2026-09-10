# DepthWizard — Implementation Bible for Coding Agent

> **WHO READS THIS:** A coding AI agent (Gemini Flash) that will implement every file.
> **RULE:** Follow these instructions EXACTLY. Do NOT improvise, skip steps, or make design decisions. Every file, every import, every function signature is specified. Just write the code and run the commands.
>
> **PROJECT ROOT:** `E:/rishabh/sih/sih-2026-problem-statements-main (1)/sih-2026-problem-statements-main/depthwizard`
> **OS:** Windows 11, Shell: PowerShell
> **HARDWARE:** RTX 4060 8GB VRAM, i7-12700H, 16GB RAM
> **PYTHON:** 3.11+
> **NODE:** 18+

---

## CRITICAL RULES FOR THE IMPLEMENTING AGENT

1. **NEVER** use placeholder images, dummy data, or `# TODO` comments. Every function must be complete.
2. **NEVER** swallow exceptions. Always log them.
3. **NEVER** hardcode secrets.
4. **ALWAYS** use absolute paths as shown below. The project root is: `E:/rishabh/sih/sih-2026-problem-statements-main (1)/sih-2026-problem-statements-main/depthwizard`
5. **ALWAYS** run verification commands after each task group to confirm things work.
6. **PowerShell quirk:** Inline Python with f-strings containing `['key']` breaks in PowerShell. Write Python to .py files and execute them instead of using `python -c`.
7. **UI Design:** Dark theme only. Glassmorphism with `backdrop-blur`. HSL color scheme: slate-900 backgrounds, blue-500 accents. Inter font. NO plain HTML.
8. When creating the frontend, use **functional React components with TypeScript**. No class components.

---

## TASK 0: Create Directory Structure

Run in PowerShell:
```powershell
$root = "E:/rishabh/sih/sih-2026-problem-statements-main (1)/sih-2026-problem-statements-main/depthwizard"
New-Item -ItemType Directory -Force -Path "$root/backend/app/api"
New-Item -ItemType Directory -Force -Path "$root/backend/app/services"
New-Item -ItemType Directory -Force -Path "$root/backend/training"
New-Item -ItemType Directory -Force -Path "$root/backend/weights"
New-Item -ItemType Directory -Force -Path "$root/backend/logs"
New-Item -ItemType Directory -Force -Path "$root/backend/exports"
New-Item -ItemType Directory -Force -Path "$root/benchmark"
```

---

## TASK 1: Backend Dependencies

Create file: `backend/requirements.txt`
```
torch>=2.1.0
torchvision>=0.16.0
transformers>=4.40.0
datasets>=2.19.0
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
rasterio>=1.3.0
Pillow>=10.0.0
numpy>=1.24.0
scipy>=1.11.0
structlog>=24.1.0
```

Run:
```powershell
cd "$root/backend"
pip install -r requirements.txt
```

**VERIFY:**
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
python -c "import rasterio; print('rasterio OK')"
python -c "import fastapi; print('fastapi OK')"
```

Must show: `CUDA: True`, GPU name, and OK messages.

---

## TASK 2: Backend Config

Create file: `backend/app/__init__.py` (empty file)

Create file: `backend/app/config.py`
```python
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKEND_ROOT = Path(__file__).parent.parent
WEIGHTS_DIR = BACKEND_ROOT / "weights"
LOGS_DIR = BACKEND_ROOT / "logs"
EXPORTS_DIR = BACKEND_ROOT / "exports"

# Model config
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEVICE = "cuda"
INFERENCE_SIZE = 512  # pixels

# Training config
TRAIN_SAMPLES = 200
TRAIN_EPOCHS = 10
TRAIN_BATCH_SIZE = 8
TRAIN_LR = 5e-5
TRAIN_IMG_SIZE = 512

# Validation
MAX_UPLOAD_SIZE_MB = 100
MIN_IMAGE_DIM = 64
MAX_IMAGE_DIM = 8192
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
```

---

## TASK 3: Logging Config

Create file: `backend/app/logging_config.py`
```python
import structlog
import logging
import sys

def setup_logging():
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

log = structlog.get_logger()
```

---

## TASK 4: Depth Estimator Service

Create file: `backend/app/services/__init__.py` (empty file)

Create file: `backend/app/services/depth_estimator.py`
```python
"""
Depth Anything V2 ViT-S inference engine.
Uses HuggingFace transformers for loading pretrained weights.
Supports optional fine-tuned weight loading.
"""
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from pathlib import Path
from app.config import MODEL_ID, DEVICE, WEIGHTS_DIR
from app.logging_config import log


class DepthEstimator:
    def __init__(self):
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        log.info("Loading Depth Anything V2 ViT-S", device=str(self.device))

        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
        self.model.to(self.device)
        self.model.eval()

        # Try loading fine-tuned weights if they exist
        finetuned_path = WEIGHTS_DIR / "best_model.pth"
        if finetuned_path.exists():
            log.info("Loading fine-tuned weights", path=str(finetuned_path))
            state_dict = torch.load(finetuned_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            log.info("Fine-tuned weights loaded successfully")

        log.info("DepthEstimator ready",
                 params=f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")

    @torch.no_grad()
    def predict(self, rgb_image: Image.Image) -> np.ndarray:
        """
        Predict relative depth from an RGB PIL Image.

        Args:
            rgb_image: PIL Image in RGB mode

        Returns:
            depth: np.ndarray of shape (H, W), float32 in [0, 1]
        """
        original_size = rgb_image.size  # (W, H)

        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = self.model(**inputs)

        predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(original_size[1], original_size[0]),  # (H, W)
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)

        # Normalize to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-8:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        return depth

    @torch.no_grad()
    def predict_with_confidence(self, rgb_image: Image.Image, n_passes: int = 5) -> tuple:
        """
        MC Dropout inference for uncertainty estimation.
        Returns (mean_depth, confidence_map) both as (H, W) float32 arrays.
        """
        original_size = rgb_image.size
        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.model.train()  # Enable dropout
        predictions = []

        for _ in range(n_passes):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model(**inputs)
            pred = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=(original_size[1], original_size[0]),
                mode="bicubic", align_corners=False
            ).squeeze()
            predictions.append(pred)

        self.model.eval()

        stacked = torch.stack(predictions)
        mean_depth = stacked.mean(dim=0).cpu().numpy().astype(np.float32)
        variance = stacked.var(dim=0).cpu().numpy().astype(np.float32)

        # Normalize mean depth to [0, 1]
        d_min, d_max = mean_depth.min(), mean_depth.max()
        if d_max - d_min > 1e-8:
            mean_depth = (mean_depth - d_min) / (d_max - d_min)

        # Confidence = 1 - normalized variance
        v_max = variance.max()
        if v_max > 1e-8:
            confidence = 1.0 - (variance / v_max)
        else:
            confidence = np.ones_like(variance)

        return mean_depth, confidence
```

**VERIFY:** Create a quick test script and run it:
```powershell
cd "$root/backend"
python -c "
from PIL import Image
import numpy as np
img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
from app.services.depth_estimator import DepthEstimator
de = DepthEstimator()
d = de.predict(img)
print(f'Output shape: {d.shape}, range: [{d.min():.3f}, {d.max():.3f}]')
print('DEPTH ESTIMATOR OK')
"
```

---

## TASK 5: GeoReader Service

Create file: `backend/app/services/geospatial.py`
```python
"""
GeoTIFF and standard image reader.
Detects CRS, affine transform, and geospatial metadata.
"""
import io
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from app.logging_config import log

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


@dataclass
class ImageMetadata:
    width: int = 0
    height: int = 0
    channels: int = 3
    is_georef: bool = False
    crs: Optional[str] = None
    transform: Optional[list] = None  # Affine as list of 6 floats
    bounds: Optional[dict] = None
    gsd: Optional[float] = None  # meters/pixel
    format: str = "unknown"


def read_image(file_bytes: bytes, filename: str) -> tuple:
    """
    Read an image from bytes. Returns (rgb_array, pil_image, metadata).
    Supports GeoTIFF, PNG, JPG.
    """
    ext = Path(filename).suffix.lower()
    metadata = ImageMetadata(format=ext)

    if ext in {".tif", ".tiff"} and HAS_RASTERIO:
        return _read_geotiff(file_bytes, metadata)
    else:
        return _read_standard(file_bytes, metadata)


def _read_geotiff(file_bytes: bytes, metadata: ImageMetadata) -> tuple:
    """Read GeoTIFF with rasterio, extract CRS and transform."""
    with rasterio.open(io.BytesIO(file_bytes)) as ds:
        # Read RGB bands (first 3)
        bands_to_read = min(ds.count, 3)
        rgb = np.stack([ds.read(i + 1) for i in range(bands_to_read)], axis=-1)

        if bands_to_read == 1:
            rgb = np.stack([rgb.squeeze()] * 3, axis=-1)

        # Handle 16-bit images
        if rgb.dtype == np.uint16:
            rgb = (rgb / 256).astype(np.uint8)
        elif rgb.dtype == np.float32 or rgb.dtype == np.float64:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

        metadata.width = ds.width
        metadata.height = ds.height
        metadata.channels = bands_to_read

        if ds.crs is not None:
            metadata.is_georef = True
            metadata.crs = str(ds.crs)
            t = ds.transform
            metadata.transform = [t.a, t.b, t.c, t.d, t.e, t.f]
            metadata.bounds = {
                "left": ds.bounds.left, "bottom": ds.bounds.bottom,
                "right": ds.bounds.right, "top": ds.bounds.top
            }
            metadata.gsd = abs(t.a)
            log.info("GeoTIFF loaded", crs=metadata.crs, gsd=metadata.gsd)
        else:
            metadata.is_georef = False
            log.info("GeoTIFF without CRS, treating as non-georeferenced")

    pil_image = Image.fromarray(rgb)
    return rgb, pil_image, metadata


def _read_standard(file_bytes: bytes, metadata: ImageMetadata) -> tuple:
    """Read PNG/JPG with PIL."""
    pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    rgb = np.array(pil_image)

    metadata.width = pil_image.width
    metadata.height = pil_image.height
    metadata.channels = 3
    metadata.is_georef = False

    log.info("Standard image loaded", size=f"{metadata.width}x{metadata.height}")
    return rgb, pil_image, metadata
```

---

## TASK 6: Calibration Service

Create file: `backend/app/services/calibration.py`
```python
"""
Two-Component Elevation Decomposition.
DSM(x,y) = DTM_base(x,y) + α · nDSM_pred(x,y)

For MVP: uses statistical prior (no real SRTM download).
"""
import numpy as np
from dataclasses import dataclass
from app.logging_config import log


@dataclass
class CalibrationResult:
    dsm: np.ndarray           # (H, W) float32, calibrated elevation
    alpha: float               # scale factor
    dsm_min: float
    dsm_max: float
    dsm_mean: float
    mode: str                  # "metric_prior" or "relative"
    unit: str                  # "meters" or "relative"


def calibrate_depth(
    relative_depth: np.ndarray,
    is_georef: bool,
    gsd: float = None,
    target_range: float = 50.0,  # default assumed max building height in meters
) -> CalibrationResult:
    """
    Convert relative depth [0,1] to calibrated DSM.

    For georeferenced: scale to approximate metric range using GSD and statistical prior.
    For non-georeferenced: keep as relative [0,1].
    """
    if is_georef and gsd is not None:
        return _metric_calibration(relative_depth, gsd, target_range)
    else:
        return _relative_calibration(relative_depth)


def _metric_calibration(depth: np.ndarray, gsd: float, target_range: float) -> CalibrationResult:
    """
    Statistical prior calibration.
    Assumes urban scene with buildings up to target_range meters.
    α = target_range / depth_range
    """
    # Ground level = 5th percentile
    ground = np.percentile(depth, 5)
    ndsm = np.maximum(depth - ground, 0.0)

    # Scale factor from statistical prior
    depth_range = np.percentile(ndsm, 99) - np.percentile(ndsm, 1)
    if depth_range < 1e-6:
        alpha = 1.0
    else:
        alpha = target_range / depth_range

    # Base elevation (simulated DTM at 0m for MVP)
    dtm_base = 0.0
    dsm = dtm_base + alpha * ndsm

    log.info("Metric calibration", alpha=f"{alpha:.2f}",
             dsm_range=f"[{dsm.min():.1f}, {dsm.max():.1f}]m")

    return CalibrationResult(
        dsm=dsm.astype(np.float32),
        alpha=float(alpha),
        dsm_min=float(dsm.min()),
        dsm_max=float(dsm.max()),
        dsm_mean=float(dsm.mean()),
        mode="metric_prior",
        unit="meters"
    )


def _relative_calibration(depth: np.ndarray) -> CalibrationResult:
    """Keep as relative depth normalized to [0, 1]."""
    log.info("Relative calibration (non-georeferenced)")
    return CalibrationResult(
        dsm=depth.astype(np.float32),
        alpha=1.0,
        dsm_min=float(depth.min()),
        dsm_max=float(depth.max()),
        dsm_mean=float(depth.mean()),
        mode="relative",
        unit="relative"
    )
```

---

## TASK 7: Mesh Builder Service

Create file: `backend/app/services/mesh_builder.py`
```python
"""
Heightfield mesh generation with mathematically correct vertices, faces, UVs, normals.
Outputs mesh data as numpy arrays and a 16-bit PNG heightmap for GPU displacement.
"""
import numpy as np
from PIL import Image
import io
import base64
from app.logging_config import log


def build_mesh_data(
    dsm: np.ndarray,
    rgb: np.ndarray,
    max_grid: int = 512,
    vertical_scale: float = 1.0,
    pixel_size: float = 1.0,
) -> dict:
    """
    Build heightfield mesh data from DSM.

    Returns dict with:
      - heightmap_b64: base64 16-bit PNG for GPU displacement
      - rgb_b64: base64 JPEG of the RGB texture
      - mesh_stats: {width, height, vertices, triangles, elevation_range}
      - dsm_colorized_b64: base64 PNG of Turbo-colormapped DSM
    """
    H, W = dsm.shape

    # Decimate if needed (for WebGL performance)
    if H > max_grid or W > max_grid:
        from scipy.ndimage import zoom
        scale_h = max_grid / H
        scale_w = max_grid / W
        scale = min(scale_h, scale_w)
        dsm_decimated = zoom(dsm, scale, order=1)
    else:
        dsm_decimated = dsm

    dH, dW = dsm_decimated.shape

    # === Heightmap as 16-bit PNG ===
    # Normalize to [0, 65535]
    d_min, d_max = dsm_decimated.min(), dsm_decimated.max()
    if d_max - d_min > 1e-8:
        hm_normalized = (dsm_decimated - d_min) / (d_max - d_min)
    else:
        hm_normalized = np.zeros_like(dsm_decimated)

    hm_16bit = (hm_normalized * 65535).astype(np.uint16)
    hm_image = Image.fromarray(hm_16bit, mode='I;16')
    hm_buf = io.BytesIO()
    hm_image.save(hm_buf, format='PNG')
    heightmap_b64 = base64.b64encode(hm_buf.getvalue()).decode()

    # === RGB texture as JPEG ===
    rgb_image = Image.fromarray(rgb)
    rgb_buf = io.BytesIO()
    rgb_image.save(rgb_buf, format='JPEG', quality=90)
    rgb_b64 = base64.b64encode(rgb_buf.getvalue()).decode()

    # === Colorized DSM (Turbo colormap) ===
    dsm_colorized = apply_turbo_colormap(hm_normalized)
    dsm_color_image = Image.fromarray(dsm_colorized)
    dsm_buf = io.BytesIO()
    dsm_color_image.save(dsm_buf, format='PNG')
    dsm_colorized_b64 = base64.b64encode(dsm_buf.getvalue()).decode()

    # === Mesh stats ===
    num_vertices = dH * dW
    num_triangles = 2 * (dH - 1) * (dW - 1)

    stats = {
        "width": dW,
        "height": dH,
        "original_width": W,
        "original_height": H,
        "vertices": num_vertices,
        "triangles": num_triangles,
        "elevation_min": float(d_min),
        "elevation_max": float(d_max),
        "elevation_range": float(d_max - d_min),
        "pixel_size": pixel_size,
    }

    log.info("Mesh built", **stats)

    return {
        "heightmap_b64": heightmap_b64,
        "rgb_b64": rgb_b64,
        "dsm_colorized_b64": dsm_colorized_b64,
        "mesh_stats": stats,
        "dsm_raw": dsm.tolist(),  # Full resolution DSM for cross-section tool
    }


def apply_turbo_colormap(values: np.ndarray) -> np.ndarray:
    """
    Apply Turbo colormap to normalized [0,1] values.
    Returns (H, W, 3) uint8 RGB array.
    """
    # Simplified Turbo colormap LUT (256 entries)
    import colorsys

    H, W = values.shape
    flat = values.flatten()

    # Generate turbo-like colormap: blue → cyan → green → yellow → red
    lut = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.25:
            r, g, b = 0.0, t * 4, 1.0
        elif t < 0.5:
            r, g, b = 0.0, 1.0, 1.0 - (t - 0.25) * 4
        elif t < 0.75:
            r, g, b = (t - 0.5) * 4, 1.0, 0.0
        else:
            r, g, b = 1.0, 1.0 - (t - 0.75) * 4, 0.0
        lut[i] = [int(r * 255), int(g * 255), int(b * 255)]

    indices = (flat * 255).astype(np.uint8)
    colorized = lut[indices].reshape(H, W, 3)
    return colorized
```

---

## TASK 8: Validation Service

Create file: `backend/app/services/validation.py`
```python
"""Compute RMSE, MAE, Pearson r between predicted and reference DSMs."""
import numpy as np
from dataclasses import dataclass


@dataclass
class ValidationMetrics:
    rmse: float
    mae: float
    pearson_r: float
    delta_1: float  # % pixels within 1.25x of GT
    n_pixels: int


def compute_metrics(pred: np.ndarray, ref: np.ndarray) -> ValidationMetrics:
    """
    Compute accuracy metrics between predicted and reference DSMs.
    Both must be (H, W) float arrays with same shape.
    """
    assert pred.shape == ref.shape, f"Shape mismatch: {pred.shape} vs {ref.shape}"

    # Mask invalid pixels
    valid = (ref > 0.01) & np.isfinite(pred) & np.isfinite(ref)
    p = pred[valid]
    r = ref[valid]
    n = p.size

    if n == 0:
        return ValidationMetrics(rmse=0, mae=0, pearson_r=0, delta_1=0, n_pixels=0)

    # RMSE
    rmse = float(np.sqrt(np.mean((p - r) ** 2)))

    # MAE
    mae = float(np.mean(np.abs(p - r)))

    # Pearson correlation
    if np.std(p) > 1e-8 and np.std(r) > 1e-8:
        pearson_r = float(np.corrcoef(p, r)[0, 1])
    else:
        pearson_r = 0.0

    # Delta accuracy (% within 1.25x)
    ratio = np.maximum(p / (r + 1e-8), r / (p + 1e-8))
    delta_1 = float(np.mean(ratio < 1.25) * 100)

    return ValidationMetrics(
        rmse=rmse, mae=mae, pearson_r=pearson_r,
        delta_1=delta_1, n_pixels=n
    )
```

---

## TASK 9: API Schemas

Create file: `backend/app/api/__init__.py` (empty file)

Create file: `backend/app/api/schemas.py`
```python
from pydantic import BaseModel
from typing import Optional


class HealthResponse(BaseModel):
    status: str
    gpu: str
    vram_gb: float


class InferenceResponse(BaseModel):
    request_id: str
    heightmap_b64: str
    rgb_b64: str
    dsm_colorized_b64: str
    mesh_stats: dict
    calibration: dict
    dsm_raw: list  # Full resolution DSM for frontend tools
    is_georef: bool
    inference_time_ms: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
```

---

## TASK 10: API Routes

Create file: `backend/app/api/routes.py`
```python
import uuid
import time
import io
import numpy as np
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from app.config import MAX_UPLOAD_SIZE_MB, ALLOWED_EXTENSIONS, EXPORTS_DIR
from app.api.schemas import InferenceResponse, HealthResponse
from app.logging_config import log
import torch

router = APIRouter()

# These get set by main.py on startup
depth_estimator = None


def set_estimator(estimator):
    global depth_estimator
    depth_estimator = estimator


@router.get("/health", response_model=HealthResponse)
async def health():
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    else:
        gpu = "CPU"
        vram = 0
    return HealthResponse(status="ok", gpu=gpu, vram_gb=round(vram, 2))


@router.post("/upload", response_model=InferenceResponse)
async def upload_image(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())[:8]
    t_start = time.time()

    log.info("Upload received", request_id=request_id, filename=file.filename)

    # 1. Validate file
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format '{ext}'. Accepted: {ALLOWED_EXTENSIONS}")

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f}MB). Max: {MAX_UPLOAD_SIZE_MB}MB")

    # 2. Read image
    from app.services.geospatial import read_image
    try:
        rgb, pil_image, metadata = read_image(file_bytes, file.filename)
    except Exception as e:
        log.error("Failed to read image", error=str(e))
        raise HTTPException(422, f"Failed to read image: {str(e)}")

    # 3. Depth inference
    from app.services.depth_estimator import DepthEstimator
    relative_depth = depth_estimator.predict(pil_image)

    # 4. Calibration
    from app.services.calibration import calibrate_depth
    cal_result = calibrate_depth(
        relative_depth,
        is_georef=metadata.is_georef,
        gsd=metadata.gsd,
    )

    # 5. Build mesh data
    from app.services.mesh_builder import build_mesh_data
    mesh_data = build_mesh_data(
        dsm=cal_result.dsm,
        rgb=rgb,
        pixel_size=metadata.gsd or 1.0,
    )

    # 6. Cache DSM for export
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EXPORTS_DIR / f"{request_id}_dsm.npy", cal_result.dsm)
    np.save(EXPORTS_DIR / f"{request_id}_rgb.npy", rgb)

    # Save metadata for export
    import json
    meta_dict = {
        "is_georef": metadata.is_georef,
        "crs": metadata.crs,
        "transform": metadata.transform,
        "width": metadata.width,
        "height": metadata.height,
    }
    with open(EXPORTS_DIR / f"{request_id}_meta.json", "w") as f:
        json.dump(meta_dict, f)

    inference_time = (time.time() - t_start) * 1000

    log.info("Inference complete", request_id=request_id,
             time_ms=f"{inference_time:.0f}",
             dsm_range=f"[{cal_result.dsm_min:.1f}, {cal_result.dsm_max:.1f}]")

    return InferenceResponse(
        request_id=request_id,
        heightmap_b64=mesh_data["heightmap_b64"],
        rgb_b64=mesh_data["rgb_b64"],
        dsm_colorized_b64=mesh_data["dsm_colorized_b64"],
        mesh_stats=mesh_data["mesh_stats"],
        calibration={
            "alpha": cal_result.alpha,
            "min": cal_result.dsm_min,
            "max": cal_result.dsm_max,
            "mean": cal_result.dsm_mean,
            "mode": cal_result.mode,
            "unit": cal_result.unit,
        },
        dsm_raw=mesh_data["dsm_raw"],
        is_georef=metadata.is_georef,
        inference_time_ms=round(inference_time, 1),
    )


@router.get("/export/{request_id}")
async def export_dsm(request_id: str):
    """Download computed DSM as GeoTIFF."""
    dsm_path = EXPORTS_DIR / f"{request_id}_dsm.npy"
    meta_path = EXPORTS_DIR / f"{request_id}_meta.json"

    if not dsm_path.exists():
        raise HTTPException(404, "DSM not found. Run inference first.")

    dsm = np.load(dsm_path)

    import json
    with open(meta_path) as f:
        meta = json.load(f)

    output_path = EXPORTS_DIR / f"{request_id}_dsm.tif"

    import rasterio
    from rasterio.transform import Affine

    if meta["transform"]:
        transform = Affine(*meta["transform"])
        crs = meta["crs"]
    else:
        transform = Affine(1.0, 0, 0, 0, -1.0, dsm.shape[0])
        crs = None

    with rasterio.open(
        str(output_path), 'w', driver='GTiff',
        height=dsm.shape[0], width=dsm.shape[1],
        count=1, dtype='float32',
        crs=crs, transform=transform,
    ) as dst:
        dst.write(dsm, 1)

    return FileResponse(
        str(output_path),
        media_type="image/tiff",
        filename=f"depthwizard_dsm_{request_id}.tif"
    )
```

---

## TASK 11: FastAPI Main App

Create file: `backend/app/main.py`
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.logging_config import setup_logging, log
from app.api.routes import router, set_estimator
from app.services.depth_estimator import DepthEstimator


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    log.info("DepthWizard starting up...")

    estimator = DepthEstimator()
    set_estimator(estimator)

    log.info("DepthWizard ready!")
    yield
    log.info("DepthWizard shutting down.")


app = FastAPI(
    title="DepthWizard API",
    description="Single-View Height Estimation & 3D Flythrough — SIH26175",
    version="0.1.0-mvp",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")
```

**VERIFY BACKEND:**
```powershell
cd "$root/backend"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
# In another terminal:
# curl http://localhost:8000/api/health
# Should return {"status":"ok","gpu":"NVIDIA GeForce RTX 4060","vram_gb":8.0}
```

Stop the server after verifying.

---

## TASK 12: Training — Loss Functions

Create file: `backend/training/losses.py`
```python
"""
Scale-Invariant Logarithmic Loss (SILog) + Gradient Matching Loss.
References: Eigen 2014, Depth Anything V2.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SILogLoss(nn.Module):
    def __init__(self, lambd: float = 0.5, eps: float = 1e-3):
        super().__init__()
        self.lambd = lambd
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target > self.eps
        if valid.sum() < 10:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_valid = pred[valid].clamp(min=self.eps)
        target_valid = target[valid]

        d = torch.log(pred_valid) - torch.log(target_valid)
        n = d.numel()

        loss = (d ** 2).sum() / n - self.lambd * (d.sum() ** 2) / (n ** 2)
        return loss


class GradientMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
            target = target.unsqueeze(0).unsqueeze(0)
        elif pred.dim() == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)

        pred_dx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_dy = F.conv2d(pred, self.sobel_y, padding=1)
        target_dx = F.conv2d(target, self.sobel_x, padding=1)
        target_dy = F.conv2d(target, self.sobel_y, padding=1)

        return (pred_dx - target_dx).abs().mean() + (pred_dy - target_dy).abs().mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.silog = SILogLoss()
        self.grad = GradientMatchingLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        return self.silog(pred, target) + self.alpha * self.grad(pred, target)
```

---

## TASK 13: Training — Dataset

Create file: `backend/training/dataset_gamus.py`
```python
"""PyTorch Dataset wrapping HuggingFace earthflow/GAMUS."""
from torch.utils.data import Dataset
from datasets import load_dataset
import torchvision.transforms as T
import torch
import numpy as np
from PIL import Image


class GAMUSDataset(Dataset):
    def __init__(self, split: str = "train", max_samples: int = 200, img_size: int = 512):
        print(f"Loading GAMUS dataset (split={split}, max_samples={max_samples})...")
        full = load_dataset("earthflow/GAMUS", split=split)
        n = min(max_samples, len(full))
        self.data = full.select(range(n))
        self.img_size = img_size
        print(f"Loaded {n} samples.")

        self.rgb_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.depth_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Handle different column names
        if "image" in sample:
            rgb = sample["image"]
        elif "rgb" in sample:
            rgb = sample["rgb"]
        else:
            rgb = sample[list(sample.keys())[0]]

        if "annotation" in sample:
            depth = sample["annotation"]
        elif "ndsm" in sample:
            depth = sample["ndsm"]
        elif "depth" in sample:
            depth = sample["depth"]
        else:
            depth = sample[list(sample.keys())[1]]

        if not isinstance(rgb, Image.Image):
            rgb = Image.fromarray(np.array(rgb))
        if not isinstance(depth, Image.Image):
            depth = Image.fromarray(np.array(depth))

        rgb = rgb.convert("RGB")
        depth = depth.convert("L")

        rgb_tensor = self.rgb_transform(rgb)
        depth_tensor = self.depth_transform(depth)
        depth_tensor = depth_tensor.clamp(min=0.0)

        return rgb_tensor, depth_tensor.squeeze(0)
```

---

## TASK 14: Training Script

Create file: `backend/training/train_gamus.py`
```python
"""
DepthWizard MVP Training Script.
Fine-tunes Depth Anything V2 ViT-S on 200 GAMUS samples.
Expected time: ~2 minutes on RTX 4060.
"""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import time
import json
from pathlib import Path
from transformers import AutoModelForDepthEstimation

from training.losses import CombinedLoss
from training.dataset_gamus import GAMUSDataset
from app.config import (
    MODEL_ID, WEIGHTS_DIR, LOGS_DIR,
    TRAIN_SAMPLES, TRAIN_EPOCHS, TRAIN_BATCH_SIZE, TRAIN_LR
)


def train():
    assert torch.cuda.is_available(), "CUDA required for training!"
    device = torch.device("cuda:0")
    torch.backends.cudnn.benchmark = True

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # 1. Dataset
    dataset = GAMUSDataset(split="train", max_samples=TRAIN_SAMPLES)
    loader = DataLoader(
        dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )

    # 2. Model
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
    model.to(device)
    model.train()

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_EPOCHS)
    scaler = GradScaler()
    criterion = CombinedLoss(alpha=0.5).to(device)

    # 4. Logging
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / f"training_mvp.jsonl"

    best_loss = float("inf")
    total_start = time.time()

    print(f"\nStarting training: {len(loader)} batches/epoch, {TRAIN_EPOCHS} epochs\n")

    for epoch in range(1, TRAIN_EPOCHS + 1):
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast(dtype=torch.float16):
                outputs = model(pixel_values=images)
                pred = outputs.predicted_depth

                # Resize prediction to match target
                if pred.shape[-2:] != targets.shape[-2:]:
                    pred = nn.functional.interpolate(
                        pred.unsqueeze(1), size=targets.shape[-2:],
                        mode="bilinear", align_corners=False
                    ).squeeze(1)

                # Normalize pred to similar range as target
                pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                target_norm = (targets - targets.min()) / (targets.max() - targets.min() + 1e-8)

                loss = criterion(pred_norm, target_norm)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # Log every batch
            log_entry = {
                "epoch": epoch, "batch": batch_idx,
                "loss": round(loss.item(), 5),
                "lr": optimizer.param_groups[0]["lr"],
            }
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{TRAIN_EPOCHS}] Loss: {avg_loss:.4f} Time: {elapsed:.1f}s")

        # Save checkpoint at epoch 5 and 10
        if epoch % 5 == 0:
            ckpt_path = WEIGHTS_DIR / f"checkpoint_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  → Checkpoint saved: {ckpt_path}")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), WEIGHTS_DIR / "best_model.pth")
            print(f"  → Best model updated (loss: {best_loss:.4f})")

    # Save final
    torch.save(model.state_dict(), WEIGHTS_DIR / "final_model.pth")

    total_time = (time.time() - total_start) / 60
    print(f"\n{'='*50}")
    print(f"Training complete in {total_time:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Weights saved to: {WEIGHTS_DIR}")
    print(f"Logs saved to: {log_file}")


if __name__ == "__main__":
    train()
```

**RUN TRAINING:**
```powershell
cd "$root/backend"
python training/train_gamus.py
```

Expected: ~2 minutes, loss decreasing, `weights/best_model.pth` created.

---

## TASK 15: Frontend — Scaffold

Run these commands:
```powershell
cd "$root"
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install three @types/three @react-three/fiber @react-three/drei
npm install recharts
npm install lucide-react
npm install tailwindcss @tailwindcss/vite
npm install d3-contour d3-geo
npm install @types/d3-contour @types/d3-geo
```

**IMPORTANT:** After scaffold, the implementing agent must create/modify all the frontend files listed in tasks 16–26 below. Every file must be complete.

---

## TASK 16–26: Frontend Components

> **INSTRUCTION TO IMPLEMENTING AGENT:**
> Create ALL of the following frontend files. Each file should be a complete, working React TypeScript component. The UI design must be:
> - **Dark theme** (slate-900 bg, slate-800 panels)
> - **Glassmorphism** (backdrop-blur-xl on panels, bg-white/5 or bg-white/10, subtle borders)
> - **Accent color:** blue-500 (#3b82f6)
> - **Font:** Inter (import from Google Fonts in index.html)
> - **Smooth transitions:** All panels animate in/out
> - **NO default/plain HTML** — everything must be styled
>
> The frontend should have these states:
> 1. `idle` — Show dropzone for file upload
> 2. `uploading` — Show progress indicator
> 3. `processing` — Show processing animation
> 4. `viewing` — Show results: DualViewer + TerrainCanvas + UVP tools
>
> **API base URL:** `http://localhost:8000/api`

### Key Frontend Files to Create:

1. **`src/main.tsx`** — React entry point, renders `<App />`
2. **`src/App.tsx`** — Main app with state management (idle/uploading/processing/viewing), layout
3. **`src/styles/globals.css`** — Tailwind directives + custom dark glassmorphism tokens
4. **`src/components/Dropzone.tsx`** — Drag-and-drop upload zone with dashed border, hover glow, calls POST /api/upload
5. **`src/components/DualViewer.tsx`** — Side-by-side RGB (left) vs colorized DSM (right), renders base64 images from API response
6. **`src/components/TerrainCanvas.tsx`** — **THE MAIN 3D COMPONENT.** Uses `@react-three/fiber` and `@react-three/drei`. Creates a `PlaneGeometry(width, height, 512, 512)` mesh. Uses the heightmap as displacement map and RGB as color texture. Implements WASD + mouse flythrough controls using drei's `PointerLockControls` or `FlyControls`. Adds fog and lighting. Accepts `verticalScale` prop.
7. **`src/components/FloodSimulator.tsx`** — Range slider for water level. Renders a semi-transparent blue plane at the water level height in the 3D scene. Shows "X% area flooded" stat.
8. **`src/components/CrossSection.tsx`** — Two-click mode: user clicks 2 points on terrain via raycaster → sample 256 elevation values along the line → render Recharts `<LineChart>` showing elevation profile (X=distance, Y=elevation).
9. **`src/components/ContourOverlay.tsx`** — Uses `d3-contour` to compute contour lines from the DSM data → renders as `THREE.Line` objects on the terrain. Toggle button + interval slider.
10. **`src/components/ColorBar.tsx`** — Vertical gradient bar showing min→max elevation values with the colormap.
11. **`src/components/SettingsPanel.tsx`** — Glassmorphic panel with controls: vertical exaggeration slider, fog toggle, contour toggle, active tool selector (measure/flood/cross-section).
12. **`src/hooks/useInference.ts`** — Custom hook: `useInference()` returns `{ upload, data, loading, error }`. Calls POST /api/upload with FormData.
13. **`src/utils/colormap.ts`** — Turbo colormap function for JS side.

### Three.js Terrain Implementation Details:

The TerrainCanvas MUST work as follows:
1. Receive `heightmapB64` and `rgbB64` from the API response
2. Create textures from these base64 strings using `THREE.TextureLoader` or `THREE.DataTexture`
3. For the heightmap: decode the base64 PNG, create a `DataTexture` with `FloatType`
4. Create `PlaneGeometry` with sufficient subdivisions (512×512)
5. Use `MeshStandardMaterial` with `displacementMap` set to the heightmap texture
6. Set `displacementScale` based on elevation range and vertical exaggeration
7. Add `PointerLockControls` or custom WASD controls for flythrough
8. Add `Fog` with exponential falloff
9. Add directional light + ambient light for terrain shading

### Flood Simulator Implementation:
- Create a `Mesh` with `PlaneGeometry` positioned at waterLevel height
- Semi-transparent blue material (`MeshStandardMaterial` with `opacity: 0.6`, `transparent: true`, color `#0077be`)
- The plane clips to the terrain area
- A slider controls the height, updating the plane's Y position in real-time

### Cross-Section Implementation:
- Toggle "cross-section mode" button
- In this mode, clicks on terrain are captured via `Raycaster`
- First click sets point A, second click sets point B
- Sample 256 points along the line AB using bilinear interpolation on the DSM data
- Render a `<LineChart>` from Recharts in a glassmorphic bottom panel
- Draw the line on the terrain using `THREE.Line`

### Contour Lines Implementation:
- Import `contours` from `d3-contour`
- Feed the DSM array into `contours().size([w, h]).thresholds(levels)`
- Convert the resulting GeoJSON-like paths to Three.js `Line` geometries
- Position them at the correct elevation on the terrain
- Toggle visibility from settings panel

---

## TASK 27: Update Frontend Config Files

### `vite.config.ts`
```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
})
```

### `index.html` — Add Inter font
```html
<!doctype html>
<html lang="en" class="dark">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/favicon.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <title>DepthWizard — Single-View Height Estimation</title>
  </head>
  <body class="bg-slate-950 text-white font-[Inter]">
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

---

## TASK 28: run_local.bat

Create file: `depthwizard/run_local.bat`
```bat
@echo off
echo.
echo  ============================================
echo   DepthWizard - Single-View Height Estimation
echo   SIH26175 - ISRO SAC
echo  ============================================
echo.
echo Starting backend server...
start /B cmd /c "cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo Backend starting on http://localhost:8000
echo.
timeout /t 5 /nobreak >nul
echo Starting frontend...
cd frontend && npm run dev
```

---

## TASK 29: README.md

Create file: `depthwizard/README.md`
```markdown
# DepthWizard 🧙‍♂️

**Single-View Height Estimation & 3D Flythrough**
SIH26175 — ISRO Space Applications Centre

## Quick Start

1. **Install backend:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Install frontend:**
   ```bash
   cd frontend
   npm install
   ```

3. **Run:**
   ```bash
   run_local.bat
   ```

4. Open http://localhost:5173

## Features
- Upload RGB satellite imagery (PNG/JPG/GeoTIFF)
- AI-powered depth estimation (Depth Anything V2)
- Scale calibration for metric elevation
- Interactive 3D terrain flythrough (WASD controls)
- 🌊 Flood inundation simulator
- ✂️ Elevation cross-section tool
- 🗺️ Contour line overlay
- 📥 GeoTIFF DSM export

## Tech Stack
- **Backend:** FastAPI + PyTorch + rasterio
- **Frontend:** Vite + React + Three.js + Tailwind
- **Model:** Depth Anything V2 ViT-S (25M params)
- **Training:** SILog + Gradient Matching loss on GAMUS dataset
```

---

## TASK 30: Final Verification Checklist

After ALL files are created and both servers are running:

```powershell
# 1. Backend health check
curl http://localhost:8000/api/health
# Expected: {"status":"ok","gpu":"NVIDIA GeForce RTX 4060","vram_gb":8.0}

# 2. Upload test image
# Use any satellite/aerial image PNG
curl -X POST http://localhost:8000/api/upload -F "file=@benchmark/sample.png"
# Expected: JSON with heightmap_b64, rgb_b64, dsm_colorized_b64, mesh_stats

# 3. Frontend loads
# Open http://localhost:5173
# Expected: Dark glassmorphic UI with dropzone

# 4. End-to-end flow
# Drop an image → wait for processing → see:
#   - DualViewer (RGB vs colorized DSM)
#   - 3D terrain with flythrough
#   - Flood slider works
#   - Cross-section works
#   - Contour lines toggle works
```

---

## TASK 31: Execution Order Summary

The implementing agent MUST execute tasks in this order:

```
TASK 0  → Create directories
TASK 1  → Install backend dependencies + VERIFY CUDA
TASK 2  → config.py
TASK 3  → logging_config.py
TASK 4  → depth_estimator.py + VERIFY inference works
TASK 5  → geospatial.py
TASK 6  → calibration.py
TASK 7  → mesh_builder.py
TASK 8  → validation.py
TASK 9  → schemas.py
TASK 10 → routes.py
TASK 11 → main.py + VERIFY backend starts and /health returns OK
TASK 12 → losses.py
TASK 13 → dataset_gamus.py
TASK 14 → train_gamus.py + RUN TRAINING + VERIFY best_model.pth exists
TASK 15 → Frontend scaffold + npm install
TASK 16–26 → ALL frontend components (implement in order listed)
TASK 27 → vite.config.ts + index.html
TASK 28 → run_local.bat
TASK 29 → README.md
TASK 30 → FULL VERIFICATION
```

---

## CONTEXT THE IMPLEMENTING AGENT NEEDS

### What is this project?
DepthWizard is an SIH 2026 hackathon entry for ISRO SAC. It takes a single satellite RGB image and:
1. Estimates depth/height using a fine-tuned Depth Anything V2 model
2. Calibrates to metric elevation
3. Renders an interactive 3D terrain flythrough
4. Provides disaster management tools (flood simulation, cross-sections)

### Key architectural decisions (DO NOT CHANGE):
- **Model:** Depth Anything V2 ViT-S from HuggingFace `depth-anything/Depth-Anything-V2-Small-hf`
- **Loss:** SILog + GradientMatching (scale-invariant, handles depth ambiguity)
- **3D:** Three.js with PlaneGeometry + displacementMap (GPU-side vertex displacement, not CPU mesh)
- **Mesh grid:** Max 512×512 vertices (262K) for 60 FPS WebGL performance
- **Calibration:** Two-component decomposition with statistical prior (no real SRTM in MVP)
- **Training:** 200 GAMUS samples, 10 epochs, batch 8, fp16 → ~2 minutes

### What makes this unique vs other SIH teams:
1. **Flood simulator** — drag slider to flood terrain (disaster management theme)
2. **Cross-section tool** — click 2 points to see elevation profile
3. **Contour lines** — cartographic isolines on 3D terrain
4. **GeoTIFF export** — standard geospatial format output
5. **SILog loss** — academically grounded, not naive MSE
