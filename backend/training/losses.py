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
        # Standardize shapes to (B, H, W)
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        batch_size = pred.shape[0]
        losses = []

        # Audit Finding 4.1 FIX: Per-sample SILog calculation eliminates cross-sample prediction bias coupling
        for b in range(batch_size):
            p = pred[b]
            t = target[b]
            valid = t > self.eps
            if valid.sum() < 10:
                # Preserve autograd connectivity
                losses.append(p.sum() * 0.0)
                continue

            pred_valid = p[valid].clamp(min=self.eps)
            target_valid = t[valid]

            d = torch.log(pred_valid) - torch.log(target_valid)
            n = d.numel()

            loss_b = (d ** 2).sum() / n - self.lambd * (d.sum() ** 2) / (n ** 2)
            losses.append(loss_b)

        if len(losses) == 0:
            return pred.sum() * 0.0
        return torch.stack(losses).mean()


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

        target = target.to(dtype=pred.dtype)
        sobel_x = self.sobel_x.to(dtype=pred.dtype)
        sobel_y = self.sobel_y.to(dtype=pred.dtype)

        pred_dx = F.conv2d(pred, sobel_x, padding=1)
        pred_dy = F.conv2d(pred, sobel_y, padding=1)
        target_dx = F.conv2d(target, sobel_x, padding=1)
        target_dy = F.conv2d(target, sobel_y, padding=1)

        return (pred_dx - target_dx).abs().mean() + (pred_dy - target_dy).abs().mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.silog = SILogLoss()
        self.grad = GradientMatchingLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        return self.silog(pred, target) + self.alpha * self.grad(pred, target)
