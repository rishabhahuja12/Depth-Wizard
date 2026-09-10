"""
Mathematically Stabilized Precision Loss for DepthWizard.
Combines:
1. StableSILogLoss (Log1p formulation eliminating near-zero gradient singularities)
2. SmoothL1Loss (Huber loss anchoring absolute elevation range without explosion)
3. GradientMatchingLoss (Sobel operators preserving razor-sharp building footprints)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StableSILogLoss(nn.Module):
    """
    Log1p Scale-Invariant Logarithmic Loss.
    d = ln(1 + gamma * p) - ln(1 + gamma * t)
    Derivative d/dx ln(1 + 10x) = 10 / (1 + 10x) in [0.91, 10.0] for x in [0, 1].
    Completely eliminates the 1/x -> infinity singularity near ground level.
    """
    def __init__(self, lambd: float = 0.5, gamma: float = 10.0, eps: float = 1e-4):
        super().__init__()
        self.lambd = lambd
        self.gamma = gamma
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 4 and pred.shape[1] == 1:
            pred = pred.squeeze(1)
        if target.dim() == 4 and target.shape[1] == 1:
            target = target.squeeze(1)

        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)

        batch_size = pred.shape[0]
        losses = []

        for b in range(batch_size):
            p = pred[b]
            t = target[b]
            valid = t > self.eps
            if valid.sum() < 25:
                losses.append(p.sum() * 0.0)
                continue

            pred_valid = p[valid].clamp(min=0.0, max=1.0)
            target_valid = t[valid].clamp(min=0.0, max=1.0)

            d = torch.log1p(self.gamma * pred_valid) - torch.log1p(self.gamma * target_valid)
            n = d.numel()

            loss_b = (d ** 2).sum() / n - self.lambd * (d.sum() ** 2) / (n ** 2)
            loss_b = torch.clamp(loss_b, min=0.0)
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
    def __init__(self, alpha_silog: float = 0.5, alpha_l1: float = 0.3, alpha_grad: float = 0.2):
        super().__init__()
        self.silog = StableSILogLoss()
        self.grad = GradientMatchingLoss()
        self.alpha_silog = alpha_silog
        self.alpha_l1 = alpha_l1
        self.alpha_grad = alpha_grad

    def forward(self, pred, target):
        loss_silog = self.silog(pred, target)
        loss_l1 = F.smooth_l1_loss(pred, target, beta=0.01)
        loss_grad = self.grad(pred, target)
        return (
            self.alpha_silog * loss_silog +
            self.alpha_l1 * loss_l1 +
            self.alpha_grad * loss_grad
        )
