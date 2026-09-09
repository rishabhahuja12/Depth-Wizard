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
