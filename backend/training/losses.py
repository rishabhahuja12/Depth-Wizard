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
        valid = target > self.eps
        if valid.sum() < 10:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_valid = pred[valid].clamp(min=self.eps)
        target_valid = target[valid]

        d = torch.log(pred_valid) - torch.log(target_valid)
        n = d.numel()

        loss = (d ** 2).sum() / n - self.lambd * (d.sum() ** 2) / (n ** 2)
        return loss


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

        pred_dx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_dy = F.conv2d(pred, self.sobel_y, padding=1)
        target_dx = F.conv2d(target, self.sobel_x, padding=1)
        target_dy = F.conv2d(target, self.sobel_y, padding=1)

        return (pred_dx - target_dx).abs().mean() + (pred_dy - target_dy).abs().mean()


class CombinedLoss(nn.Module):
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.silog = SILogLoss()
        self.grad = GradientMatchingLoss()
        self.alpha = alpha

    def forward(self, pred, target):
        return self.silog(pred, target) + self.alpha * self.grad(pred, target)
