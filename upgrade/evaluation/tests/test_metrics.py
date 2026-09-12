"""
Known-answer tests for the P0 height-evaluation metrics.

Run directly (no pytest dependency, matching the repo's existing test style):
    .venv/Scripts/python.exe upgrade/evaluation/tests/test_metrics.py

Each test hand-builds tiny arrays whose correct metric value is computable by
hand, so a regression in the math is caught immediately.
"""
import sys
from pathlib import Path

import numpy as np

# Make `metrics` importable when run from anywhere.
EVAL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EVAL_DIR))

import metrics  # noqa: E402


def _assert_close(actual, expected, tol=1e-6, label=""):
    assert abs(actual - expected) <= tol, f"{label}: expected {expected}, got {actual}"


def test_mae_and_rmse_zero_on_perfect_prediction():
    ref = np.array([[0.0, 5.0], [10.0, 20.0]], dtype=np.float32)
    pred = ref.copy()
    _assert_close(metrics.mae(pred, ref), 0.0, label="MAE perfect")
    _assert_close(metrics.rmse(pred, ref), 0.0, label="RMSE perfect")


def test_mae_and_rmse_known_values():
    # errors: [1, 1, 1, 3]  ->  MAE = 1.5,  RMSE = sqrt((1+1+1+9)/4) = sqrt(3) = 1.7320508
    ref = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    pred = np.array([11.0, 21.0, 31.0, 43.0], dtype=np.float32)
    _assert_close(metrics.mae(pred, ref), 1.5, tol=1e-5, label="MAE known")
    _assert_close(metrics.rmse(pred, ref), np.sqrt(3.0), tol=1e-5, label="RMSE known")


def test_mae_ignores_nodata_pixels():
    # The -9999 nodata pixel must be excluded, leaving a perfect match -> MAE 0.
    ref = np.array([10.0, 20.0, -9999.0], dtype=np.float32)
    pred = np.array([10.0, 20.0, 500.0], dtype=np.float32)
    _assert_close(metrics.mae(pred, ref), 0.0, label="MAE nodata-masked")


def test_delta_thresholds_known_fractions():
    # ratios = [1.0, 1.2, 1.5, 3.0]
    # d1 (<1.25):   1.0,1.2            -> 0.5
    # d2 (<1.5625): 1.0,1.2,1.5        -> 0.75
    # d3 (<1.9531): 1.0,1.2,1.5        -> 0.75
    ref = np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float32)
    pred = np.array([10.0, 12.0, 15.0, 30.0], dtype=np.float32)
    d1, d2, d3 = metrics.delta_thresholds(pred, ref)
    _assert_close(d1, 0.5, label="delta1")
    _assert_close(d2, 0.75, label="delta2")
    _assert_close(d3, 0.75, label="delta3")


def test_si_rmse_zero_under_pure_scale():
    # SI-RMSE is scale-invariant: a pure global scale must give ~0.
    ref = np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float32)
    pred = ref * 2.0
    _assert_close(metrics.si_rmse(pred, ref), 0.0, tol=1e-5, label="SI-RMSE pure scale")


def test_si_rmse_positive_when_shape_differs():
    ref = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
    pred = np.array([1.0, 3.0, 4.0, 5.0], dtype=np.float32)  # not a pure scale
    assert metrics.si_rmse(pred, ref) > 1e-3, "SI-RMSE should be >0 when shape differs"


def test_affine_align_recovers_linear_transform():
    # A relative-depth model output that is an affine transform of truth should,
    # after best-fit scale+shift alignment, match truth almost exactly.
    ref = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    pred = 0.3 * ref + 7.0  # arbitrary scale + shift (the "relative" case)
    aligned_rmse = metrics.affine_aligned_rmse(pred, ref)
    _assert_close(aligned_rmse, 0.0, tol=1e-4, label="affine-aligned RMSE")


def test_affine_fit_maps_prediction_onto_reference():
    # affine_fit returns (s, t) minimizing ||s*pred + t - ref||; applying it to
    # a linear-transform prediction must reproduce the reference.
    ref = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    pred = 0.3 * ref + 7.0
    s, t = metrics.affine_fit(pred, ref)
    aligned = s * pred + t
    assert np.allclose(aligned, ref, atol=1e-4), f"aligned={aligned} != ref={ref}"


def test_boundary_f_perfect_on_matching_edge():
    ref = np.zeros((8, 8), dtype=np.float32)
    ref[:, 4:] = 10.0  # vertical step edge
    pred = ref.copy()
    f = metrics.boundary_f_score(pred, ref, edge_threshold=1.0)
    _assert_close(f, 1.0, tol=1e-6, label="boundary-F matching edge")


def test_boundary_f_zero_when_prediction_is_flat():
    ref = np.zeros((8, 8), dtype=np.float32)
    ref[:, 4:] = 10.0
    pred = np.zeros((8, 8), dtype=np.float32)  # no edges at all
    f = metrics.boundary_f_score(pred, ref, edge_threshold=1.0)
    _assert_close(f, 0.0, tol=1e-6, label="boundary-F flat prediction")


def test_boundary_f_tolerates_small_edge_shift():
    # A 2px-shifted edge should score high with tolerance, ~0 without it.
    ref = np.zeros((16, 16), dtype=np.float32); ref[:, 8:] = 10.0   # edge at col 8
    pred = np.zeros((16, 16), dtype=np.float32); pred[:, 10:] = 10.0  # edge at col 10
    f_tol = metrics.boundary_f_score(pred, ref, edge_threshold=1.0, tol=2)
    f_exact = metrics.boundary_f_score(pred, ref, edge_threshold=1.0, tol=0)
    assert f_tol > 0.9, f"2px-shifted edge should match within tol=2, got {f_tol}"
    assert f_exact < 0.1, f"exact match should collapse on a 2px shift, got {f_exact}"


def test_tall_structure_mae_only_counts_tall_pixels():
    # ref > 15 only at the last two pixels; error there is [2, 4] -> MAE 3.
    # The short-pixel errors (huge) must be ignored.
    ref = np.array([1.0, 2.0, 20.0, 40.0], dtype=np.float32)
    pred = np.array([100.0, 100.0, 22.0, 44.0], dtype=np.float32)
    _assert_close(metrics.tall_structure_mae(pred, ref, height_threshold=15.0), 3.0,
                  label="tall-structure MAE")


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
        else:
            passed += 1
            print(f"ok    {t.__name__}")
    print(f"\n{passed}/{len(tests)} passed")
    return passed == len(tests)


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
