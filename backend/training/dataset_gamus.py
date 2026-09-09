"""PyTorch Dataset wrapping HuggingFace earthflow/GAMUS."""
from torch.utils.data import Dataset
from datasets import load_dataset
import torchvision.transforms as T
import torch
import numpy as np
from PIL import Image


class GAMUSDataset(Dataset):
    def __init__(self, split: str = "train", max_samples: int = 200, img_size: int = 512):
        print(f"Loading GAMUS dataset (split={split}, max_samples={max_samples})...")
        full = load_dataset("earthflow/GAMUS", split=split)
        n = min(max_samples, len(full))
        self.data = full.select(range(n))
        self.img_size = img_size
        print(f"Loaded {n} samples.")

        self.rgb_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.depth_transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Handle different column names
        if "image" in sample:
            rgb = sample["image"]
        elif "rgb" in sample:
            rgb = sample["rgb"]
        else:
            rgb = sample[list(sample.keys())[0]]

        if "annotation" in sample:
            depth = sample["annotation"]
        elif "ndsm" in sample:
            depth = sample["ndsm"]
        elif "depth" in sample:
            depth = sample["depth"]
        else:
            depth = sample[list(sample.keys())[1]]

        if not isinstance(rgb, Image.Image):
            rgb = Image.fromarray(np.array(rgb))
        if not isinstance(depth, Image.Image):
            depth = Image.fromarray(np.array(depth))

        rgb = rgb.convert("RGB")
        depth = depth.convert("L")

        rgb_tensor = self.rgb_transform(rgb)
        depth_tensor = self.depth_transform(depth)
        depth_tensor = depth_tensor.clamp(min=0.0)

        return rgb_tensor, depth_tensor.squeeze(0)
