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
