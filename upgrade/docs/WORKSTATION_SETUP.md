# Training Workstation Setup (RTX 4500 Ada, 24 GB)

The dev box authors code; **this workstation runs the heavy work** — the full-split
P0 baseline and all P1 training. Do this once, then keep it in sync via `git pull`.

> **The one trap:** `pip install torch` installs a **CPU-only** build on Windows.
> You MUST install the CUDA build from PyTorch's wheel index (step 4), or the GPU
> sits idle. Verify with step 5 before trusting anything.

---

## 1. Get the code
```powershell
git clone https://github.com/rishabhahuja12/Depth-Wizard.git
cd Depth-Wizard
git checkout research/depth-model-upgrade
git lfs install && git lfs pull        # pulls best_model.pth (284 MB)
```
(Already cloned? Just `git checkout research/depth-model-upgrade && git pull`.)

## 2. Check the GPU + its CUDA version
```powershell
nvidia-smi
```
Note the **CUDA Version** shown top-right (e.g. 12.4). Pick the matching torch wheel
tag below: 12.1→`cu121`, 12.4→`cu124`, 12.6→`cu126`. Pick the closest wheel **at or
below** your driver's CUDA version.

## 3. Python 3.11 virtual environment
Use **3.11** (not 3.13 — rasterio/torch wheels are unreliable on 3.13).
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

## 4. Install the CUDA build of torch FIRST
```powershell
# replace cu124 with the tag from step 2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
Then the rest of the deps (this won't downgrade torch):
```powershell
pip install -r requirements.txt
```

## 5. Verify CUDA is actually live (do not skip)
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```
Expect something like: `2.x.x+cu124 True NVIDIA RTX 4500 Ada ...`
If it prints `+cpu` or `False`, redo step 4 — do not proceed.

## 6. (Optional) HuggingFace token for faster GAMUS downloads
The `test` split is ~2,861 tile pairs streamed from HF; a token lifts rate limits.
```powershell
setx HF_TOKEN "hf_your_token_here"      # then open a new shell
```

---

## 7. Run the authoritative P0 baseline
Smoke-test the environment first (few tiles, downloads the model once):
```powershell
.venv\Scripts\python.exe upgrade\evaluation\run_baseline.py --limit 5
```
Then the full held-out split — this is the committable BEFORE table:
```powershell
.venv\Scripts\python.exe upgrade\evaluation\run_baseline.py
```
Outputs land in `upgrade\outputs\`:
- `baseline_report.md` — the metric tables (overall + per-city)
- `baseline_metrics.csv` — machine-readable

> Full run = large HF download (tens of GB, cached under `upgrade\data\`) + one
> forward pass per tile on the GPU. Budget accordingly; it only needs to run once
> per baseline. Re-running is cheap after the download is cached.

## 8. Still open before the numbers are final
- **GAMUS GSD = 0.33 m** — VERIFIED (arXiv:2305.14914, "resolution of 0.33m"); it is
  now the `--gsd` default, no action needed.
- **Boundary-F sanity** — came out ~0 on the dev smoke run (blurry ViT-S edges as
  expected, but the edge threshold deserves one look on real GPU output).

---

## 9. Run P1 Stage 1 — metric fine-tune (after the P0 baseline exists)
First a CPU wiring check (no GPU/GAMUS, uses the cached small model):
```powershell
.venv\Scripts\python.exe upgrade\training\train_metric.py --smoke
```
Then the real Stage-1 fine-tune on the GPU (downloads DA2-Large ~1.3 GB once,
streams the GAMUS train split, early-stops on held-out val MAE):
```powershell
.venv\Scripts\python.exe upgrade\training\train_metric.py
```
- Output checkpoint: `upgrade\checkpoints\metric_best.pth` (+ its held-out MAE).
- VRAM tight? drop to `--batch 2 --grad-accum 8`.
- **Kill-gate:** Stage-1 (SiLog+Smooth-L1 in meters) must beat the P0 baseline MAE
  before adding edge (Stage 2) or long-tail (Stage 3). If it doesn't, debug data/
  scaling first — do NOT stack more losses on a broken base.

## Sanity: unit tests run anywhere (CPU fine), no GPU needed
```powershell
.venv\Scripts\python.exe upgrade\evaluation\tests\test_metrics.py
.venv\Scripts\python.exe upgrade\evaluation\tests\test_loader.py
```
