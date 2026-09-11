# SETUP — Workstation Gentle Weekend Run (P0 baseline + P1 Stage 1)

Self-contained guide for the training workstation (RTX 4500 Ada, 24 GB). Tuned to
run **gently over the weekend**: **low learning rate**, **GPU kept under ~50%**
(both memory and utilization), trading speed for a cool, half-idle card.

> This supersedes the training section of `WORKSTATION_SETUP.md`; the env steps
> below are the same, the run config is the gentle one.

---

## 1. Code
```powershell
git clone https://github.com/rishabhahuja12/Depth-Wizard.git
cd Depth-Wizard
git checkout research/depth-model-upgrade
git lfs install && git lfs pull
```
(Already cloned: `git checkout research/depth-model-upgrade && git pull`.)

## 2. GPU + CUDA version
```powershell
nvidia-smi
```
Note the **CUDA Version** (top-right). Wheel tag: 12.1→`cu121`, 12.4→`cu124`, 12.6→`cu126`
(closest **at or below** your driver's CUDA).

## 3. Python 3.11 venv
```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

## 4. Install CUDA torch FIRST (the one trap — plain `pip install torch` is CPU-only)
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # match step 2
pip install -r requirements.txt
```

## 5. Verify CUDA is live (do not skip)
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Must print `...+cu1xx True NVIDIA RTX 4500 Ada ...`. If `+cpu`/`False`, redo step 4.

---

## 6. Cap the GPU BEFORE training (keeps it under ~50%)

**(a) Power limit** — the cleanest ceiling on sustained compute/heat. Read the max,
then set roughly half (needs an elevated shell):
```powershell
nvidia-smi -q -d POWER | Select-String "Max Power Limit"     # e.g. 210 W
nvidia-smi -pl 105                                           # ~50% of max
```
(If `-pl` is refused, skip it — the throttle in step 7 still holds utilization down.)

**(b) VRAM** — hard-capped in-process to 50% by the `--max-vram-frac 0.5` flag below
(+ batch 2 + gradient checkpointing), so the run stays under ~12 GB of 24 GB.

**(c) Utilization** — the `--throttle-sleep` flag idles briefly each step so average
GPU utilization stays low. Start at 0.15 s and tune (step 9).

---

## 7. P0 baseline first (the BEFORE table)
Smoke, then full:
```powershell
.venv\Scripts\python.exe upgrade\evaluation\run_baseline.py --limit 5
.venv\Scripts\python.exe upgrade\evaluation\run_baseline.py
```
Outputs → `upgrade\outputs\baseline_report.md` + `baseline_metrics.csv`.

---

## 8. P1 Stage 1 — gentle metric fine-tune

Wiring check (CPU, no GPU/GAMUS):
```powershell
.venv\Scripts\python.exe upgrade\training\train_metric.py --smoke
```

The weekend run — **low LR, small batch, VRAM-capped, throttled** — logged to a file
so it survives a disconnect:
```powershell
.venv\Scripts\python.exe upgrade\training\train_metric.py `
  --epochs 50 --warmup 3 `
  --enc-lr 2.5e-6 --head-lr 2.5e-5 `
  --batch 2 --grad-accum 8 --crop 518 `
  --max-vram-frac 0.5 --throttle-sleep 0.15 `
  --workers 4 *> upgrade\outputs\train_stage1.log
```
What each choice does:
| Flag | Value | Why |
|---|---|---|
| `--enc-lr / --head-lr` | 2.5e-6 / 2.5e-5 | **Half the standard low LRs** — gentle, steady; we have time |
| `--epochs / --warmup` | 50 / 3 | Low LR needs more epochs; longer warmup = softer start |
| `--batch / --grad-accum` | 2 / 8 | eff. batch 16 but tiny VRAM footprint |
| `--max-vram-frac` | 0.5 | hard cap → process can't exceed ~12 GB |
| `--throttle-sleep` | 0.15 | idles each step → holds average GPU utilization down |

Output checkpoint: `upgrade\checkpoints\metric_best.pth` (+ its held-out val MAE).
First epoch is slow (streams/caches the GAMUS train split); later epochs are faster.
Rough budget: ~20–30 h for 50 throttled epochs — comfortably a weekend.

---

## 9. Watch it stay under 50% (in a second shell)
```powershell
while ($true) { nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader; Start-Sleep 5 }
```
- **VRAM** should sit well under ~12 GB. If it OOMs, drop `--crop 448` or `--batch 1`.
- **Utilization** should hover ≤ 50%. If it runs hotter, **raise `--throttle-sleep`**
  (e.g. 0.25–0.4); if it's crawling far below 50%, lower it toward 0.

---

## 10. The kill-gate (why we measured first)
Compare `train_stage1.log`'s best **val MAE** against the P0 baseline MAE:
- **Beats baseline** → proceed to Stage 2 (add the Sobel edge loss), then Stage 3.
- **Doesn't beat it** → debug data/scaling BEFORE stacking more losses. Don't pile
  loss terms on a base that isn't winning.

Bring both numbers back and we'll decide Stage 2 together.

---

## Sanity: unit tests (CPU, no GPU)
```powershell
.venv\Scripts\python.exe upgrade\evaluation\tests\test_metrics.py
.venv\Scripts\python.exe upgrade\training\tests\test_metric_losses.py
.venv\Scripts\python.exe upgrade\training\tests\test_metric_dataset.py
.venv\Scripts\python.exe upgrade\training\tests\test_train_helpers.py
```
