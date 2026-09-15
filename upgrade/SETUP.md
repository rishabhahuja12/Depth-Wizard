# SETUP — Workstation Training Run (P0 baseline + P1 Stages 1–3)

Self-contained guide for the training workstation (RTX 4500 Ada, 24 GB). Tuned to
run **gently**: **low learning rate**, **GPU kept under ~50%** (both memory and
utilization), trading speed for a cool, half-idle card.

> Tiles are streamed directly from the HuggingFace Hub mirror (`earthflow/GAMUS`)
> and cached locally in `upgrade/data/gamus_cache/` on first use. **No prefetch
> step is needed** — just run training and tiles download on-demand.

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
(If `-pl` is refused, skip it — `--throttle-sleep` in the training command still
holds utilization down.)

**(b) VRAM** — hard-capped in-process to 50% by the `--max-vram-frac 0.5` flag
(+ batch 2 + gradient checkpointing), so the run stays under ~12 GB of 24 GB.

**(c) Utilization** — the `--throttle-sleep` flag idles briefly each step so average
GPU utilization stays low. Start at 0.15 s and tune (step 9).

---

## 7. P0 baseline (the BEFORE table)
Smoke, then full. (Uses the `test` split — streams a small number of tiles from HF.)
```powershell
.venv\Scripts\python.exe upgrade\evaluation\run_baseline.py --limit 5
.venv\Scripts\python.exe upgrade\evaluation\run_baseline.py
```
Outputs → `upgrade\outputs\baseline_report.md` + `baseline_metrics.csv`.

---

## 8. Sanity: unit tests (CPU, no GPU) — all should pass before trusting a run
```powershell
Get-ChildItem -Recurse upgrade -Filter test_*.py | ForEach-Object {
  .venv\Scripts\python.exe $_.FullName
}
```

---

## 9. P1 Stage 1 — metric fine-tune

Wiring check (CPU, no GPU, no data download):
```powershell
.venv\Scripts\python.exe upgrade\training\train_metric.py --smoke
```

The full run — **low LR, small batch, VRAM-capped, throttled** — logged to a file so
it survives a disconnect. Tiles are fetched from HF Hub on-demand and cached locally;
re-runs reuse the cache:
```powershell
.venv\Scripts\python.exe upgrade\training\train_metric.py --resume `
  --epochs 50 --warmup 3 `
  --enc-lr 2.5e-6 --head-lr 2.5e-5 `
  --batch 2 --grad-accum 8 --crop 504 `
  --max-vram-frac 0.5 --throttle-sleep 0.15 `
  --workers 4 *>> upgrade\outputs\train_stage1.log
```

What each flag does:

| Flag | Value | Why |
|---|---|---|
| `--resume` | on | continue from `metric_last.pth` if interrupted (safe to re-run) |
| `--enc-lr / --head-lr` | 2.5e-6 / 2.5e-5 | **Half the standard low LRs** — gentle, steady |
| `--epochs / --warmup` | 50 / 3 | Low LR needs more epochs; longer warmup = softer start |
| `--batch / --grad-accum` | 2 / 8 | eff. batch 16 but tiny VRAM footprint |
| `--crop` | 504 | 36×14 — a clean DINOv2 patch multiple; 448 (32×14) is the OOM fallback |
| `--max-vram-frac` | 0.5 | hard cap → process can't exceed ~12 GB |
| `--throttle-sleep` | 0.15 | idles each step → holds average GPU utilization ≤ 50% |

> **Tile caching:** the first pass downloads each tile from `earthflow/GAMUS` on
> HuggingFace and writes it to `upgrade\data\gamus_cache\`. All subsequent runs
> (including resumed ones) read from this local cache — no repeated downloads.

Checkpoints (`upgrade\checkpoints\`): `metric_best.pth` (best val MAE) and
`metric_last.pth` (full state for resume). **Interrupted?** Just re-run the exact
same command — `--resume` picks up from the last completed epoch. `*>>` appends to
the log so resumed runs don't clobber history.

Rough budget: ~20–30 h for 50 throttled epochs.

---

## 10. Watch it stay under 50% (in a second shell)
```powershell
while ($true) { nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader; Start-Sleep 5 }
```
- **VRAM** should sit well under ~12 GB. If it OOMs, drop `--crop 448` or `--batch 1`.
- **Utilization** should hover ≤ 50%. If it runs hotter, **raise `--throttle-sleep`**
  (e.g. 0.25–0.4); if it's crawling far below 50%, lower it toward 0.

---

## 11. The kill-gate (why we measured first)
Compare `train_stage1.log`'s best **val MAE** against the P0 baseline MAE:
- **Beats baseline** → proceed to Stage 2 (§12).
- **Doesn't beat it** → debug data/scaling BEFORE stacking more losses.

Bring both numbers back and we'll decide together.

---

## 12. Later stages — ONLY after the previous one wins

Everything below is coded and unit-tested; run in this **gated order** — each step
must beat the previous on val MAE before the next.

**Stage 2 — add the edge (Sobel) loss** (sharper building/ridge walls):
```powershell
.venv\Scripts\python.exe upgrade\training\train_metric.py --resume `
  --w-silog 1.0 --w-l1 1.0 --w-grad 0.5 `
  --epochs 60 --crop 504 --max-vram-frac 0.5 --throttle-sleep 0.15 `
  --workers 4 *>> upgrade\outputs\train_stage2.log
```

**Stage 3 — add long-tail (tall-structure) reweighting** (only if Stage 2 helped):
```powershell
.venv\Scripts\python.exe upgrade\training\train_metric.py --resume `
  --w-silog 1.0 --w-l1 1.0 --w-grad 0.5 --w-lt 0.5 `
  --epochs 60 --crop 504 --max-vram-frac 0.5 --throttle-sleep 0.15 `
  --workers 4 *>> upgrade\outputs\train_stage3.log
```

### Optional: LoRA instead of full fine-tune (fallback / for testing Giant)
Needs `pip install peft`. Frees VRAM, trains faster, slight ceiling cost:
```powershell
.venv\Scripts\python.exe upgrade\training\train_metric.py --resume --lora `
  --lora-r 16 --lora-alpha 32 --epochs 50 --crop 504 `
  --max-vram-frac 0.5 --throttle-sleep 0.15 *>> upgrade\outputs\train_lora.log
```

---

## 13. Dataset blend (building-height aux) — after GAMUS-only wins

Adopted flow, gated one source at a time on val MAE:

  GAMUS → **+Open-Canopy** (forest) → **+M4Heights** (urban, HF-easy) → **+DFC2023**
  (global diversity) → **NL pair-builder** (0.5 m LiDAR quality ceiling)

All adapters are wired; see `docs/DATASETS_GUIDE.md` and `docs/DATASET_LICENSES.md`
for per-source download and licence details.

```powershell
.venv\Scripts\python.exe upgrade\training\train_metric.py --resume --blend `
  --oc-root upgrade\data\open_canopy `
  --aux-fraction 0.3 --aux-gsd 0.5 --crop 504 `
  --max-vram-frac 0.5 --throttle-sleep 0.15 *>> upgrade\outputs\train_blend.log
```

---

## Known limitations

- **Windows + HDF5 workers:** if `--workers 4` causes `PermissionError`/IO stutter
  on cached h5 tiles, drop to `--workers 2` or `0`.
- **16-bit RGB** is percentile-stretched per channel for viewability; colours may
  shift slightly and it costs a little CPU per tile.
- **Off-nadir flag** can false-positive on strong regular grids (Manhattan, farmland);
  it's a warning, not a correction.
- **VRAM cap** (`--max-vram-frac 0.5`) hard-errors on OOM rather than paging — drop
  `--crop`/`--batch` if it hits.
- **First-epoch latency:** the first pass through the dataset downloads all tiles from
  HF Hub; subsequent epochs and resumed runs read from the local cache and are fast.
