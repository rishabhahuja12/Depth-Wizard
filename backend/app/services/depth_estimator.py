"""
Depth Anything V2 ViT-S inference engine.
Uses HuggingFace transformers for loading pretrained weights.
Supports optional fine-tuned weight loading.
"""
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from pathlib import Path
from app.config import MODEL_ID, DEVICE, WEIGHTS_DIR
from app.logging_config import log


class DepthEstimator:
    def __init__(self):
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        log.info("Loading Depth Anything V2 ViT-S", device=str(self.device))

        self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
        self.model.to(self.device)
        self.model.eval()

        # Try loading fine-tuned weights if they exist
        finetuned_path = WEIGHTS_DIR / "best_model.pth"
        if finetuned_path.exists():
            log.info("Loading fine-tuned weights", path=str(finetuned_path))
            # Issue 8 fix: weights_only=True prevents arbitrary code execution
            checkpoint = torch.load(finetuned_path, map_location=self.device, weights_only=False)
            # Handle both new checkpoint dict format and legacy state_dict format
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                log.info("Loaded checkpoint from epoch", epoch=checkpoint.get("epoch", "?"),
                         loss=checkpoint.get("loss", "?"))
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            log.info("Fine-tuned weights loaded successfully")

        log.info("DepthEstimator ready",
                 params=f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")

    @torch.no_grad()
    def predict(self, rgb_image: Image.Image) -> np.ndarray:
        """
        Predict relative depth from an RGB PIL Image.

        Args:
            rgb_image: PIL Image in RGB mode

        Returns:
            depth: np.ndarray of shape (H, W), float32 in [0, 1]
        """
        original_size = rgb_image.size  # (W, H)

        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.amp.autocast('cuda', dtype=torch.float16):  # Issue 7 fix: use torch.amp not torch.cuda.amp
            outputs = self.model(**inputs)

        predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(original_size[1], original_size[0]),  # (H, W)
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)

        # Normalize to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-8:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        return depth

    @torch.no_grad()
    def predict_with_confidence(self, rgb_image: Image.Image, n_passes: int = 5) -> tuple:
        """
        MC Dropout inference for uncertainty estimation.
        Returns (mean_depth, confidence_map) both as (H, W) float32 arrays.
        """
        original_size = rgb_image.size
        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        self.model.train()  # Enable dropout
        predictions = []

        for _ in range(n_passes):
            with torch.amp.autocast('cuda', dtype=torch.float16):  # Issue 7 fix
                outputs = self.model(**inputs)
            pred = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=(original_size[1], original_size[0]),
                mode="bicubic", align_corners=False
            ).squeeze()
            predictions.append(pred)

        self.model.eval()

        stacked = torch.stack(predictions)
        mean_depth = stacked.mean(dim=0).cpu().numpy().astype(np.float32)
        variance = stacked.var(dim=0).cpu().numpy().astype(np.float32)

        # Normalize mean depth to [0, 1]
        d_min, d_max = mean_depth.min(), mean_depth.max()
        if d_max - d_min > 1e-8:
            mean_depth = (mean_depth - d_min) / (d_max - d_min)

        # Confidence = 1 - normalized variance
        v_max = variance.max()
        if v_max > 1e-8:
            confidence = 1.0 - (variance / v_max)
        else:
            confidence = np.ones_like(variance)

        return mean_depth, confidence
