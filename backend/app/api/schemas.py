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
    normal_map_b64: Optional[str] = None
    dsm_colorized_b64: str
    mesh_stats: dict
    calibration: dict
    dsm_raw: list  # Full resolution DSM for frontend tools
    is_georef: bool
    crs: Optional[str] = None
    confidence_mean: Optional[float] = None
    inference_time_ms: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
