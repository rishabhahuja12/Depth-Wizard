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
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
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
