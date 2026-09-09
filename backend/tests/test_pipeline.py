"""Test core pipeline services: depth estimation, calibration, and mesh building."""
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.depth_estimator import DepthEstimator
from app.services.calibration import calibrate_depth
from app.services.mesh_builder import build_mesh_data
from app.services.validation import compute_metrics

def test_full_pipeline():
    print("Testing DepthEstimator loading & inference...")
    estimator = DepthEstimator()

    # Create synthetic RGB test image
    synthetic_rgb = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
    pil_img = Image.fromarray(synthetic_rgb)

    # 1. Depth prediction
    depth = estimator.predict(pil_img)
    print(f"Predicted depth shape: {depth.shape}, min: {depth.min():.3f}, max: {depth.max():.3f}")
    assert depth.shape == (256, 256)
    assert 0.0 <= depth.min() <= depth.max() <= 1.0

    # 2. Calibration
    cal = calibrate_depth(depth, is_georef=True, gsd=0.5, target_range=30.0)
    print(f"Calibrated DSM range: [{cal.dsm_min:.2f}m, {cal.dsm_max:.2f}m], alpha: {cal.alpha:.2f}")
    assert cal.dsm.shape == (256, 256)
    assert cal.unit == "meters"

    # 3. Mesh Building
    mesh_data = build_mesh_data(cal.dsm, synthetic_rgb, max_grid=256)
    print(f"Mesh stats: {mesh_data['mesh_stats']}")
    assert "heightmap_b64" in mesh_data
    assert "rgb_b64" in mesh_data
    assert "dsm_colorized_b64" in mesh_data
    assert mesh_data["mesh_stats"]["vertices"] == 256 * 256

    # 4. Validation metrics
    metrics = compute_metrics(cal.dsm, cal.dsm + np.random.normal(0, 0.5, cal.dsm.shape).astype(np.float32))
    print(f"Synthetic test metrics: RMSE={metrics.rmse:.2f}m, Pearson_r={metrics.pearson_r:.3f}")
    assert metrics.rmse > 0
    assert metrics.pearson_r > 0.9

    print("FULL PIPELINE VERIFICATION PASSED!")

if __name__ == "__main__":
    test_full_pipeline()
