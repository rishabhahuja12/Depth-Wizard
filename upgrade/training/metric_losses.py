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


class MetricLoss(nn.Module):
    """Stage-1 composite: weighted SILog + Smooth-L1, all in meters."""

    def __init__(self, w_silog: float = 1.0, w_l1: float = 1.0,
                 lambd: float = 0.85, beta: float = 1.0):
        super().__init__()
        self.w_silog = w_silog
        self.w_l1 = w_l1
        self.silog = SILogMetric(lambd=lambd)
        self.l1 = SmoothL1Metric(beta=beta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = pred.sum() * 0.0
        if self.w_silog:
            total = total + self.w_silog * self.silog(pred, target)
        if self.w_l1:
            total = total + self.w_l1 * self.l1(pred, target)
        return total
