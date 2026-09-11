"""
Known-answer tests for the P1 metric losses.

The whole point of P1 is training in METERS, not normalized [0,1]. These tests
pin the loss behaviour on hand-computable inputs so the un-normalized math can't
silently regress.

Run:
    .venv/Scripts/python.exe upgrade/training/tests/test_metric_losses.py
"""
import sys
from pathlib import Path

import torch

TRAIN_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TRAIN_DIR))

import metric_losses as ml  # noqa: E402


def _close(a, b, tol=1e-5, label=""):
    a = float(a)
    assert abs(a - b) <= tol, f"{label}: expected {b}, got {a}"


def test_silog_zero_on_perfect_prediction():
    t = torch.tensor([[1.0, 5.0, 10.0, 20.0]])
    loss = ml.SILogMetric()(t.clone(), t.clone())
    _close(loss, 0.0, label="SILog perfect")


def test_silog_pure_scale_gives_lambda_residual():
    # d = ln(2) constant -> mean(d^2) - lambd*mean(d)^2 = (1 - lambd) * ln(2)^2
    lambd = 0.85
    t = torch.tensor([[1.0, 2.0, 3.0, 4.0, 8.0]])
    pred = t * 2.0
    expected = (1.0 - lambd) * (torch.log(torch.tensor(2.0)) ** 2).item()
    loss = ml.SILogMetric(lambd=lambd)(pred, t)
    _close(loss, expected, tol=1e-5, label="SILog pure-scale")


def test_silog_ignores_ground_and_nodata_pixels():
    # target has a 0 (ground) and a -9999 (nodata); both excluded, rest perfect -> 0
    t = torch.tensor([[0.0, 10.0, -9999.0, 20.0]])
    pred = torch.tensor([[7.0, 10.0, 3.0, 20.0]])  # wrong at the excluded pixels
    loss = ml.SILogMetric()(pred, t)
    _close(loss, 0.0, label="SILog masks ground/nodata")


def test_smooth_l1_zero_on_perfect_prediction():
    t = torch.tensor([[0.0, 5.0, 10.0]])
    loss = ml.SmoothL1Metric()(t.clone(), t.clone())
    _close(loss, 0.0, label="SmoothL1 perfect")


def test_smooth_l1_known_value_in_meters():
    # every error = 2.0 m, beta=1.0 -> huber = |2| - 0.5 = 1.5 each -> mean 1.5
    t = torch.tensor([[0.0, 4.0, 10.0, 20.0]])
    pred = t + 2.0
    loss = ml.SmoothL1Metric(beta=1.0)(pred, t)
    _close(loss, 1.5, label="SmoothL1 known (meters)")


def test_smooth_l1_ignores_nodata():
    t = torch.tensor([[5.0, 10.0, -9999.0]])
    pred = torch.tensor([[5.0, 10.0, 999.0]])  # huge error only at nodata
    loss = ml.SmoothL1Metric()(pred, t)
    _close(loss, 0.0, label="SmoothL1 masks nodata")


def test_combined_weight_selects_component():
    t = torch.tensor([[1.0, 4.0, 10.0, 20.0]])
    pred = t + 2.0
    l1_only = ml.MetricLoss(w_silog=0.0, w_l1=1.0)(pred, t)
    silog_only = ml.MetricLoss(w_silog=1.0, w_l1=0.0)(pred, t)
    _close(l1_only, ml.SmoothL1Metric()(pred, t), label="combined = l1 when w_silog=0")
    _close(silog_only, ml.SILogMetric()(pred, t), label="combined = silog when w_l1=0")


# --- Stage 2: edge (Sobel gradient) loss ---

def test_edge_loss_zero_on_perfect_prediction():
    t = torch.zeros(1, 8, 8)
    t[:, :, 4:] = 10.0  # a vertical step
    loss = ml.EdgeGradientLoss()(t.clone(), t.clone())
    _close(loss, 0.0, label="edge perfect")


def test_edge_loss_positive_when_edge_missing():
    t = torch.zeros(1, 8, 8)
    t[:, :, 4:] = 10.0          # target has an edge
    pred = torch.zeros(1, 8, 8)  # prediction is flat -> misses the edge
    assert ml.EdgeGradientLoss()(pred, t) > 0.1, "edge loss should penalize a missing edge"


# --- Stage 3: long-tail (tall-structure) reweighting ---

def test_longtail_upweights_tall_pixels():
    # Ground pixel (0 m) has 0 error; tall pixel (30 m) has 4 m error.
    # Unweighted mean would be 2.0. Weights: ground=1, tall=clamp(1+30/15,1,5)=3.
    # Weighted = (1*0 + 3*4)/(1+3) = 3.0 > 2.0 — the tall error dominates.
    t = torch.tensor([[0.0, 30.0]])
    pred = torch.tensor([[0.0, 34.0]])
    lt = ml.LongTailWeightedL1(height_scale=15.0, max_weight=5.0)(pred, t)
    _close(lt, 3.0, tol=1e-5, label="longtail weighted (tall dominates)")


def test_longtail_equals_plain_when_all_ground():
    # With all-ground targets, weights collapse to 1 -> equals plain L1 (=error).
    t = torch.tensor([[0.0, 0.0, 0.0]])
    pred = torch.tensor([[1.0, 1.0, 1.0]])
    lt = ml.LongTailWeightedL1(height_scale=15.0)(pred, t)
    _close(lt, 1.0, tol=1e-5, label="longtail ~ plain L1 on ground")


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
