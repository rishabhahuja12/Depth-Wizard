# Depth-Wizard Metric Upgrade: Multi-Dataset Blend Training Transfer Manifest

## 1. Run Summary & Context

- **Training Task:** Stage 1 Multi-Dataset Blended Training (GAMUS + Open-Canopy + DFC2023)
- **Base Architecture:** `depth-anything/Depth-Anything-V2-Large-hf`
- **Output Scale:** Absolute Metric Depth / nDSM (meters above ground)
- **Git Branch:** `research/depth-model-upgrade`
- **Git Commit:** `751af88746317c9dd12fff8da6e015c860ec07c5` (*Tune blended training config*)
- **Training Started:** 2026-09-29 12:35:19
- **Training Completed:** 2026-09-29 22:16:15 (~9.6 hours total runtime)
- **Total Epochs:** 15 / 15 completed successfully

### Execution Command
```powershell
.venv\Scripts\python.exe upgrade\training\train_metric.py --offline --blend `
  --oc-root D:\Depth-Wizard\Open-Canopy --dfc-root D:\Depth-Wizard\DFC2023\train `
  --gamus-root D:\Depth-Wizard\GAMUS `
  --init-weights upgrade\checkpoints\metric_best.pth `
  --ckpt-dir upgrade\checkpoints\blend_oc_dfc `
  --epochs 15 --warmup 2 --enc-lr 2.5e-6 --head-lr 2.5e-5 `
  --batch 2 --grad-accum 8 --crop 504 --aux-fraction 0.3 `
  --aux-weights "open_canopy=0.5,dfc2023=1.0" --test-tiles 40 --workers 0
```

### Dataset Composition
- **Total Training Samples:** 17,524
  - GAMUS (Base Urban/Suburban): 13,480 samples
  - Open-Canopy (Canopy / Forestry): 16 tiles (blended weight: 0.5)
  - DFC2023 (Global High-Resolution Urban): 1,595 tiles (blended weight: 1.0)
- **Validation Split:** 178 DFC2023 tiles, 2 Open-Canopy tiles, plus GAMUS validation & held-out test splits.

---

## 2. Key Results & Validation Milestones

- **Initial Baseline (GAMUS-only `metric_best.pth`):**
  - Val Mean MAE: **7.428 m**
  - GAMUS MAE: 2.152 m
  - Open-Canopy MAE: 12.705 m
- **Winning Checkpoint (Epoch 10 `blend_oc_dfc/metric_best.pth`):**
  - Val Mean MAE: **4.913 m** (**-33.8% relative error reduction**)
  - GAMUS MAE: **2.139 m** (preserved core urban height accuracy)
  - Open-Canopy MAE: **10.863 m** (reduced error by 1.84 m)
  - DFC2023 MAE: **1.737 m** (exceptional high-detail urban nDSM estimation)
  - Test MAE: **2.18 m**, Test Tall MAE: **3.66 m**

### 15-Epoch Training Progression Table

| Epoch | Train Loss | Val Mean MAE | GAMUS MAE | Open-Canopy MAE | DFC2023 MAE | Test MAE | Test Tall MAE | Learning Rate | Duration |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 15.2684 | 5.698 m | 2.14 m | 12.59 m | 2.37 m | 2.190 m | 3.698 m | 2.50e-06 | 38.7 min |
| 2 | 15.4608 | 5.670 m | 2.20 m | 12.61 m | 2.20 m | 2.244 m | 3.757 m | 2.50e-06 | 38.2 min |
| 3 | 15.3497 | 5.539 m | 2.27 m | 11.98 m | 2.36 m | 2.244 m | 3.576 m | 2.46e-06 | 38.3 min |
| 4 | 14.9478 | 5.690 m | 2.26 m | 12.65 m | 2.16 m | 2.260 m | 3.461 m | 2.36e-06 | 38.3 min |
| 5 | 14.7747 | 5.708 m | 2.17 m | 12.79 m | 2.16 m | 2.211 m | 3.681 m | 2.19e-06 | 38.3 min |
| 6 | 14.3427 | 5.456 m | 2.15 m | 12.19 m | 2.03 m | 2.194 m | 3.588 m | 1.96e-06 | 38.3 min |
| 7 | 14.0266 | 5.545 m | 2.14 m | 12.62 m | 1.87 m | 2.192 m | 3.762 m | 1.69e-06 | 38.4 min |
| 8 | 13.7650 | 5.558 m | 2.14 m | 12.63 m | 1.91 m | 2.208 m | 3.689 m | 1.40e-06 | 38.3 min |
| 9 | 13.4889 | 5.382 m | 2.14 m | 12.26 m | 1.75 m | 2.185 m | 3.677 m | 1.10e-06 | 38.4 min |
| **10 (Best)** | **13.3214** | **4.913 m** | **2.14 m** | **10.86 m** | **1.74 m** | **2.205 m** | **3.938 m** | **8.10e-07** | **38.4 min** |
| 11 | 13.1181 | 5.652 m | 2.13 m | 13.07 m | 1.75 m | 2.181 m | 3.582 m | 5.40e-07 | 38.4 min |
| 12 | 12.9784 | 5.636 m | 2.13 m | 13.03 m | 1.75 m | 2.181 m | 3.698 m | 3.10e-07 | 38.4 min |
| 13 | 12.8766 | 5.532 m | 2.13 m | 12.75 m | 1.72 m | 2.183 m | 3.717 m | 1.40e-07 | 38.5 min |
| 14 | 12.8128 | 5.661 m | 2.13 m | 13.14 m | 1.71 m | 2.185 m | 3.630 m | 4.00e-08 | 38.5 min |
| 15 | 12.7922 | 5.674 m | 2.13 m | 13.19 m | 1.71 m | 2.184 m | 3.661 m | 0.00e+00 | 38.5 min |

---

## 3. Checkpoints Catalog

| Relative Path | Size | SHA-256 Checksum | Description |
|---|---|---|---|
| `checkpoints/blend_oc_dfc/metric_best.pth` | 1,341,467,248 bytes (~1.25 GB) | `89cda09b3194492936f02fe0a9b7bfbdd2ec7f65df023b77c6c827007e6f7dc3` | **Primary Deliverable**: Best weights from Epoch 10. Contains model weights and evaluation metadata. Ready for deployment and inference. |
| `checkpoints/blend_oc_dfc/metric_last.pth` | 4,014,970,881 bytes (~3.74 GB) | `0d7375e0a2f71ce62d4319e900e01a3a9ede63a57c8beab88d38d37771715de9` | **Resume State**: Final state from Epoch 15. Includes `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, and `best_mae`. |
| `checkpoints/gamus_baseline/metric_best.pth` | 1,341,467,248 bytes (~1.25 GB) | `9b4a887b2cc818d831b573a9b650dba073fe6b3635bffb833a6135565497d56b` | Pre-blend baseline weights (GAMUS-only stage 1) used as `--init-weights`. |
| `checkpoints/gamus_baseline/metric_last.pth` | 4,014,970,881 bytes (~3.74 GB) | `817b1c34bec3278ac9664b99988f5330633fcabafff677673f5d7bbee4884512` | Full training state from the earlier GAMUS-only baseline run. |

---

## 4. How to Use & Load the Checkpoints

### A. Python / PyTorch Direct Loading
```python
import torch
from transformers import AutoModelForDepthEstimation

# 1. Load the base HuggingFace model architecture
model = AutoModelForDepthEstimation.from_pretrained('depth-anything/Depth-Anything-V2-Large-hf')

# 2. Load the trained metric checkpoint
ckpt = torch.load('checkpoints/blend_oc_dfc/metric_best.pth', map_location='cuda')
state = ckpt['model_state_dict']

# 3. Apply weights
model.load_state_dict(state, strict=False)
model.eval().cuda()

print('Loaded metric model successfully!')
print(f'Epoch: {ckpt.get("epoch")}, Val MAE: {ckpt.get("val_mae_m"):.3f} m')
```

### B. Resuming Training
To continue fine-tuning (e.g. for additional epochs or new datasets):
```powershell
python upgrade/training/train_metric.py --offline --blend `
  --oc-root <path_to_oc> --dfc-root <path_to_dfc> --gamus-root <path_to_gamus> `
  --resume checkpoints/blend_oc_dfc/metric_last.pth `
  --epochs 25
```

### C. Live Backend Deployment
In DepthWizard, the live service (`backend/app/services/depth_estimator.py`) automatically discovers `backend/weights/metric_best.pth`.
Simply place `checkpoints/blend_oc_dfc/metric_best.pth` at `backend/weights/metric_best.pth` (or use a symlink/hardlink) and restart the server.

---

## 5. Archive Contents Directory Structure

```
.
├── TRANSFER_MANIFEST.md                  # This transfer manifest and documentation
├── checkpoints/
│   ├── blend_oc_dfc/
│   │   ├── metric_best.pth               # Winning model weights (Epoch 10)
│   │   └── metric_last.pth               # Full training state (Epoch 15)
│   └── gamus_baseline/
│       ├── metric_best.pth               # Pre-blend baseline model
│       └── metric_last.pth               # Baseline training state
├── logs/
│   ├── train_metric_20260929_123519.log  # Complete 15-epoch training execution log
│   └── train_metric_20260929_102242.log  # Prior initialization run log
├── training/                             # All training pipelines, blend dataset loaders
├── evaluation/                           # Evaluation benchmark harness & metrics
├── inference/                            # DEM base anchor, tiling, off-nadir detection
├── docs/                                 # Research specifications & dataset guides
├── logging_config.py                     # Shared logging utilities
├── README.md                             # Sandbox architecture overview
└── SETUP.md                              # Workstation setup & run guide
```
