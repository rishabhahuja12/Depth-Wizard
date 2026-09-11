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
(If `-pl` is refused, skip it — the throttle in step 9 still holds utilization down.)

**(b) VRAM** — hard-capped in-process to 50% by the `--max-vram-frac 0.5` flag below
(+ batch 2 + gradient checkpointing), so the run stays under ~12 GB of 24 GB.

**(c) Utilization** — the `--throttle-sleep` flag idles briefly each step so average
GPU utilization stays low. Start at 0.15 s and tune (step 10).

---

## 7. One-time prefetch — the ONLY step that needs internet
Download the whole **train + val** split **and** the DA2-Large model ONCE, so the
long training run can then go fully offline. It is **resumable** (the "dataset
checkpoint"): re-running skips whatever's already on disk, so a dropped connection
is harmless — just run it again.
```powershell
.venv\Scripts\python.exe upgrade\training\gamus_prefetch.py
```
- Caches to `upgrade\data\gamus_cache\` and writes `manifest_train.json` / `manifest_val.json`.
- Also caches the model for offline loading.
- Tens of GB — check disk space. Re-run until it prints **"All set"**.

> After this, **training (step 9) needs no internet at all** — you can unplug it.

## 8. P0 baseline (the BEFORE table)
Smoke, then full. (This uses the `test` split and streams online — a short one-shot
run, so a connection here is fine; it's the multi-day *training* we made offline.)
```powershell
.venv\Scripts\python.exe upgrade\evaluation\run_baseline.py --limit 5
.venv\Scripts\python.exe upgrade\evaluation\run_baseline.py
```
Outputs → `upgrade\outputs\baseline_report.md` + `baseline_metrics.csv`.

---

## 9. P1 Stage 1 — gentle metric fine-tune (OFFLINE)

Wiring check (CPU, no GPU/GAMUS):
```powershell
.venv\Scripts\python.exe upgrade\training\train_metric.py --smoke
```

The weekend run — **offline, low LR, small batch, VRAM-capped, throttled** — logged
to a file so it survives a disconnect. `--offline` reads only the prefetched data,
so you can literally unplug the network:
```powershell
.venv\Scripts\python.exe upgrade\training\train_metric.py --offline --resume `
  --epochs 50 --warmup 3 `
  --enc-lr 2.5e-6 --head-lr 2.5e-5 `
  --batch 2 --grad-accum 8 --crop 512 `
  --max-vram-frac 0.5 --throttle-sleep 0.15 `
  --workers 4 *>> upgrade\outputs\train_stage1.log
```
What each choice does:
| Flag | Value | Why |
|---|---|---|
| `--offline` | on | read only prefetched local data — **no internet, immune to drops** |
| `--resume` | on | continue from `metric_last.pth` if the run was interrupted (safe to re-run) |
| `--enc-lr / --head-lr` | 2.5e-6 / 2.5e-5 | **Half the standard low LRs** — gentle, steady; we have time |
| `--epochs / --warmup` | 50 / 3 | Low LR needs more epochs; longer warmup = softer start |
| `--batch / --grad-accum` | 2 / 8 | eff. batch 16 but tiny VRAM footprint |
| `--crop` | 512 | deterministic 512 tile-cells (4 per 1024 tile), zero augmentation |
| `--max-vram-frac` | 0.5 | hard cap → process can't exceed ~12 GB |
| `--throttle-sleep` | 0.15 | idles each step → holds average GPU utilization down |

Checkpoints (`upgrade\checkpoints\`): `metric_best.pth` (best val MAE) and
`metric_last.pth` (full state for resume). **Interrupted?** Just re-run the exact
same command — `--resume` picks up from the last completed epoch. `*>>` appends to
the log so resumed runs don't clobber history.
Rough budget: ~20–30 h for 50 throttled epochs — comfortably a weekend.

> Requires step 7. If a manifest is missing it errors immediately
> ("Offline mode but no manifest ... Run gamus_prefetch first") rather than hanging.

### Optional: LoRA instead of full fine-tune (fallback / for testing Giant)
Needs `pip install peft`. Frees VRAM, trains faster, slight ceiling cost — use only
if full-FT stalls or you want to A/B the Giant backbone:
```powershell
.venv\Scripts\python.exe upgrade\training\train_metric.py --offline --resume --lora `
  --lora-r 16 --lora-alpha 32 --epochs 50 --crop 512 `
  --max-vram-frac 0.5 --throttle-sleep 0.15 *>> upgrade\outputs\train_lora.log
```

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
- **Doesn't beat it** → debug data/scaling BEFORE stacking more losses. Don't pile
  loss terms on a base that isn't winning.

Bring both numbers back and we'll decide together.

---

## 12. Later stages — ONLY after the previous one wins (all built & tested)

Everything below is coded and unit-tested; run it in this **gated order** — each
step must beat the previous on val MAE (or its own metric) before the next.

**Stage 2 — add the edge (Sobel) loss** (sharper building/ridge walls):
```powershell
.venv\Scripts\python.exe upgrade\training\train_metric.py --offline --resume `
  --w-silog 1.0 --w-l1 1.0 --w-grad 0.5 `
  --epochs 60 --crop 512 --max-vram-frac 0.5 --throttle-sleep 0.15 *>> upgrade\outputs\train_stage2.log
```
**Stage 3 — add long-tail (tall-structure) reweighting** (only if Stage 2 helped):
```powershell
.venv\Scripts\python.exe upgrade\training\train_metric.py --offline --resume `
  --w-silog 1.0 --w-l1 1.0 --w-grad 0.5 --w-lt 0.5 `
  --epochs 60 --crop 512 --max-vram-frac 0.5 --throttle-sleep 0.15 *>> upgrade\outputs\train_stage3.log
```

**Dataset blend (Open-Canopy + GBH)** — only after GAMUS-only wins. Now runnable
via `--blend`.

1. Get the data (large — grab slices):
   - **Open-Canopy** — *primary aux slice* (HF, GeoTIFF, 1.5 m, open license):
     resumable slice download —
     ```powershell
     .venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'upgrade/training'); import aux_datasets as a; a.prefetch_open_canopy('upgrade/data/open_canopy', allow_patterns=None)"
     ```
     (Set `allow_patterns` to grab only a slice — the full set is ~360 GB.)
   - **GBH** — ❌ **DROPPED** (verified). Its paired RGB↔height training data is **not
     publicly available** — the imagery is PLANET PlanetScope (proprietary); only the
     derived global height product (GBA.Height, heights only) is public. So it can't
     feed our RGB→height task. **Blend Open-Canopy alone (omit `--gbh-root`).** Details
     in `upgrade\DATASET_LICENSES.md`.
2. **Confirm the real folder names** and pass them as subdirs (defaults may differ
   from the actual download): the sources take `rgb_subdir` / `height_subdir`.
3. Train blended (after Stage 1+ on GAMUS):
   ```powershell
   .venv\Scripts\python.exe upgrade\training\train_metric.py --offline --resume --blend `
     --oc-root upgrade\data\open_canopy `
     --aux-fraction 0.3 --aux-gsd 0.5 --crop 512 `
     --max-vram-frac 0.5 --throttle-sleep 0.15 *>> upgrade\outputs\train_blend.log
   ```
   (GBH dropped — Open-Canopy is the sole aux source. `--gbh-root` remains available
   only if you separately hold licensed paired PLANET rasters.)
   GAMUS keeps native tiling; aux is harmonized (GSD→common grid, meters) and mixed
   at `--aux-fraction` of GAMUS length, balanced across sources.

> Honest caveat: the aux source readers are validated against real GeoTIFFs (tests),
> but the exact Open-Canopy/GBH **folder layout must be confirmed** against your
> download, and GBH's **data license verified**, before a real blend run.

**Inference-side (P1b / P2 / P4)** — used when serving the trained model, not during
training. All tested, in `upgrade\inference\`:
- `dem_base.py` — compose absolute DSM = real DEM base + nDSM (flood-critical).
- `tiling.py` — seamless tiled hi-res inference (Export path; Live stays 1-pass).
- `off_nadir.py` — obliqueness flag ("⚠ oblique — reduced accuracy").

These integrate into the live app as a **separate, deliberate step** (logged in
`upgrade\INTEGRATIONS.md`) once a metric checkpoint has passed the gate.

---

## Sanity: unit tests (CPU, no GPU) — all should pass before trusting a run
```powershell
Get-ChildItem -Recurse upgrade -Filter test_*.py | ForEach-Object {
  .venv\Scripts\python.exe $_.FullName
}
```
The end-to-end `test_train_integration.py` actually runs the training loop + a
checkpoint resume on CPU — the closest proof the workstation path is sound.
