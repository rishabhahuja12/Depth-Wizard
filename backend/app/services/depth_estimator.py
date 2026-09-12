"""
Depth Anything V2 ViT-S inference engine.
Uses HuggingFace transformers for loading pretrained weights.
Supports optional fine-tuned weight loading.
"""
import torch
import contextlib
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from pathlib import Path
from app.config import MODEL_ID, DEVICE, WEIGHTS_DIR
from app.logging_config import log


def postprocess_depth(depth: np.ndarray, is_metric: bool) -> np.ndarray:
    """Turn a raw model prediction into the value we serve.

    - is_metric: the model outputs meters-above-ground already — keep the SCALE,
      only clean NaN/inf and clip negatives to ground. (Re-normalizing here is the
      bug that silently threw away every metric fine-tune.)
    - relative: robust percentile normalization to [0, 1] for a relative-depth model.
    """
    if is_metric:
        return np.maximum(np.nan_to_num(depth.astype(np.float32),
                                        nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    p_low, p_high = np.percentile(depth, [0.5, 99.5])
    if p_high - p_low > 1e-6:
        return np.clip((depth - p_low) / (p_high - p_low), 0.0, 1.0)
    d_min, d_max = depth.min(), depth.max()
    if d_max > d_min:
        return (depth - d_min) / (d_max - d_min + 1e-8)
    return np.zeros_like(depth)


class DepthEstimator:
    def __init__(self):
        self.device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
        log.info("Loading Depth Anything V2 ViT-S", device=str(self.device))

        try:
            self.processor = AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)
            self.model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, local_files_only=True)
        except Exception:
            self.processor = AutoImageProcessor.from_pretrained(MODEL_ID)
            self.model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
        self.model.to(self.device)
        self.model.eval()
        self.is_metric = False  # set True when a metric fine-tuned checkpoint is loaded

        # Try loading fine-tuned weights if they exist. Accept both the serving name
        # (best_model.pth) and the training output name (metric_best.pth), so a
        # metric checkpoint copied into weights/ is actually picked up.
        finetuned_path = next((WEIGHTS_DIR / n for n in ("best_model.pth", "metric_best.pth")
                               if (WEIGHTS_DIR / n).exists()), None)
        if finetuned_path is not None:
            log.info("Loading fine-tuned weights", path=str(finetuned_path))
            # Safe by default: weights_only=True blocks arbitrary code execution on
            # load. Our checkpoints are plain tensors + primitives, so this works;
            # fall back only for a legacy checkpoint the user placed themselves.
            try:
                checkpoint = torch.load(finetuned_path, map_location=self.device, weights_only=True)
            except Exception as e:  # noqa: BLE001
                log.warning("weights_only load failed; retrying unsafe load (trusted local file)",
                            error=str(e))
                checkpoint = torch.load(finetuned_path, map_location=self.device, weights_only=False)
            # Handle both new checkpoint dict format and legacy state_dict format
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                self.is_metric = bool(checkpoint.get("metric", False))
                log.info("Loaded checkpoint", epoch=checkpoint.get("epoch", "?"),
                         val_mae_m=checkpoint.get("val_mae_m", "?"), metric=self.is_metric)
            else:
                state_dict = checkpoint

            has_nans = any(torch.isnan(v).any() or torch.isinf(v).any() for v in state_dict.values())
            if has_nans:
                log.error("Corrupted checkpoint detected with NaNs! Refusing to load.")
            else:
                # Loud key-match check: strict=False can silently load almost nothing
                # if the checkpoint's architecture doesn't match (e.g. Large vs Small).
                result = self.model.load_state_dict(state_dict, strict=False)
                model_keys = set(self.model.state_dict().keys())
                matched = len(model_keys) - len(set(result.missing_keys))
                log.info("Fine-tuned weights loaded",
                         matched=f"{matched}/{len(model_keys)}",
                         missing=len(result.missing_keys), unexpected=len(result.unexpected_keys))
                if matched == 0:
                    self.is_metric = False
                    log.error("Checkpoint matched ZERO model keys — wrong architecture? "
                              "Serving the PRETRAINED model (metric flag cleared).")
                elif matched < 0.5 * len(model_keys):
                    log.error("Checkpoint matched <50% of model keys — likely an "
                              "architecture mismatch; predictions may be unreliable.")

        log.info("DepthEstimator ready",
                 params=f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")

    @torch.no_grad()
    def predict(self, rgb_image: Image.Image) -> np.ndarray:
        """
        Predict relative depth from an RGB PIL Image.

        Args:
            rgb_image: PIL Image in RGB mode

        Returns:
            depth: np.ndarray (H, W) float32 — meters-above-ground if a metric
            checkpoint is loaded (self.is_metric), else relative depth in [0, 1].
        """
        if isinstance(rgb_image, np.ndarray):
            rgb_image = Image.fromarray(rgb_image)

        original_size = rgb_image.size  # (W, H)

        inputs = self.processor(images=rgb_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        autocast_ctx = (
            torch.amp.autocast('cuda', dtype=torch.float16)
            if self.device.type == 'cuda'
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            outputs = self.model(**inputs)

        predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(original_size[1], original_size[0]),  # (H, W)
            mode="bicubic",
            align_corners=False,
        )[0, 0]

        depth = prediction.cpu().numpy().astype(np.float32)

        # Metric model -> keep meters; relative model -> normalize to [0,1].
        return postprocess_depth(depth, self.is_metric)

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
            autocast_ctx = (
                torch.amp.autocast('cuda', dtype=torch.float16)
                if self.device.type == 'cuda'
                else contextlib.nullcontext()
            )
            with autocast_ctx:
                outputs = self.model(**inputs)
            pred = torch.nn.functional.interpolate(
                outputs.predicted_depth.unsqueeze(1),
                size=(original_size[1], original_size[0]),
                mode="bicubic", align_corners=False
            )[0, 0]
            predictions.append(pred)

        self.model.eval()

        stacked = torch.stack(predictions)
        mean_depth = stacked.mean(dim=0).cpu().numpy().astype(np.float32)
        variance = stacked.var(dim=0).cpu().numpy().astype(np.float32)

        # Metric model -> keep meters; relative model -> normalize to [0,1].
        mean_depth = postprocess_depth(mean_depth, self.is_metric)

        # Confidence = 1 - normalized variance
        v_max = variance.max()
        if v_max > 1e-8:
            confidence = 1.0 - (variance / v_max)
        else:
            confidence = np.ones_like(variance)

        return mean_depth, confidence
