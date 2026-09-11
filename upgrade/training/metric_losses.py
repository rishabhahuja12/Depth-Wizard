"""
P1 metric losses — trained in METERS, never normalized to [0,1].

This is the deliberate opposite of backend/training/losses.py, which clamps
everything to [0,1] and so can only ever learn relative depth. Here:

  * SILogMetric  — scale-invariant log loss on the nDSM structure (target > eps),
    so the network learns correct relative shape without a scale singularity.
  * SmoothL1Metric — Huber loss in meters over all valid pixels (ground included),
    which is what actually anchors absolute height. No [0,1] clamp anywhere.
  * MetricLoss — the Stage-1 composite (SILog + Smooth-L1). Edge (Sobel) and
    long-tail terms are added in later stages behind their own weights.

All losses ignore nodata via the same range convention as the eval harness.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_NODATA_LO = -9000.0
_NODATA_HI = 15000.0


def _finite_range_mask(x: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(x) & (x > _NODATA_LO) & (x < _NODATA_HI)


class SILogMetric(nn.Module):
    """Scale-invariant log loss: mean(d^2) - lambd * mean(d)^2, d = log(pred) - log(target).
    Evaluated only where target > eps (structure), so ground/nodata don't blow up log()."""

    def __init__(self, lambd: float = 0.85, eps: float = 1e-3):
        super().__init__()
        self.lambd = lambd
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = _finite_range_mask(target) & (target > self.eps) & torch.isfinite(pred)
        if mask.sum() < 2:
            return pred.sum() * 0.0
        p = pred[mask].clamp(min=self.eps)
        t = target[mask].clamp(min=self.eps)
        d = torch.log(p) - torch.log(t)
        loss = torch.mean(d ** 2) - self.lambd * (torch.mean(d) ** 2)
        return loss.clamp(min=0.0)


class SmoothL1Metric(nn.Module):
    """Huber loss in meters over all valid (finite, in-range) pixels."""

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = _finite_range_mask(target) & torch.isfinite(pred)
        if mask.sum() < 1:
            return pred.sum() * 0.0
        return F.smooth_l1_loss(pred[mask], target[mask], beta=self.beta)


class EdgeGradientLoss(nn.Module):
    """Stage-2: Sobel gradient matching in meters — sharpens building/ridge edges.
    L1 between the Sobel responses of prediction and target. Unlike the backend
    version, no [0,1] clamp: it operates on metric heights directly."""

    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer("sx", sx)
        self.register_buffer("sy", sy)

    def _to_bchw(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return x.unsqueeze(0).unsqueeze(0)
        if x.dim() == 3:
            return x.unsqueeze(1)
        return x

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.nan_to_num(self._to_bchw(pred).float())
        t = torch.nan_to_num(self._to_bchw(target).float())
        sx, sy = self.sx.to(p.dtype), self.sy.to(p.dtype)
        pdx, pdy = F.conv2d(p, sx, padding=1), F.conv2d(p, sy, padding=1)
        tdx, tdy = F.conv2d(t, sx, padding=1), F.conv2d(t, sy, padding=1)
        return (pdx - tdx).abs().mean() + (pdy - tdy).abs().mean()


class LongTailWeightedL1(nn.Module):
    """Stage-3 (HTC-style): L1 where tall-structure pixels are up-weighted, so
    abundant ground pixels don't drown the gradient on the rare tall buildings
    (the documented tall-structure underestimation). weight = clamp(1 + t/scale,
    1, max_weight)."""

    def __init__(self, height_scale: float = 15.0, max_weight: float = 5.0):
        super().__init__()
        self.height_scale = height_scale
        self.max_weight = max_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = _finite_range_mask(target) & torch.isfinite(pred)
        if mask.sum() < 1:
            return pred.sum() * 0.0
        p, t = pred[mask].float(), target[mask].float()
        w = torch.clamp(1.0 + t / self.height_scale, min=1.0, max=self.max_weight)
        return (w * (p - t).abs()).sum() / w.sum()


class MetricLoss(nn.Module):
    """Staged composite, all in meters. Stage 1 = SILog + Smooth-L1 (w_grad,
    w_lt = 0); Stage 2 adds the edge term; Stage 3 adds long-tail reweighting.
    Staging is controlled purely by the weights."""

    def __init__(self, w_silog: float = 1.0, w_l1: float = 1.0,
                 w_grad: float = 0.0, w_lt: float = 0.0,
                 lambd: float = 0.85, beta: float = 1.0,
                 height_scale: float = 15.0, max_weight: float = 5.0):
        super().__init__()
        self.w_silog, self.w_l1, self.w_grad, self.w_lt = w_silog, w_l1, w_grad, w_lt
        self.silog = SILogMetric(lambd=lambd)
        self.l1 = SmoothL1Metric(beta=beta)
        self.grad = EdgeGradientLoss()
        self.lt = LongTailWeightedL1(height_scale=height_scale, max_weight=max_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = pred.sum() * 0.0
        if self.w_silog:
            total = total + self.w_silog * self.silog(pred, target)
        if self.w_l1:
            total = total + self.w_l1 * self.l1(pred, target)
        if self.w_grad:
            total = total + self.w_grad * self.grad(pred, target)
        if self.w_lt:
            total = total + self.w_lt * self.lt(pred, target)
        return total
