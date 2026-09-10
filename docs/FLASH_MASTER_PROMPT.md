# Master Prompt for Gemini Flash — DepthWizard Implementation

> **Copy everything below this line and paste it as your prompt to Flash.**
> **Do NOT modify anything.**

---

## YOUR MISSION

You are building **DepthWizard** — an AI-powered single-view height estimation and 3D flythrough application for SIH 2026 (ISRO SAC). The complete specification is split across two files in this workspace. You MUST read both files before writing any code.

**DEADLINE:** Everything must be working by end of today.

---

## STEP 0: READ THE SPECIFICATION FILES (DO THIS FIRST)

Before writing ANY code, read these two files completely:

1. **`E:/rishabh/sih/sih-2026-problem-statements-main (1)/sih-2026-problem-statements-main/DEPTHWIZARD_IMPLEMENTATION_BIBLE.md`**
   — Contains: Project structure, all backend Python code (FastAPI, PyTorch, depth estimation, calibration, mesh builder, training), exact pip install commands, verification steps.

2. **`E:/rishabh/sih/sih-2026-problem-statements-main (1)/sih-2026-problem-statements-main/DEPTHWIZARD_FRONTEND_BIBLE.md`**
   — Contains: All frontend React/TypeScript code (Vite, Three.js 3D terrain, flood simulator, cross-section tool, contour lines, UI components), exact npm install commands.

These files contain COMPLETE source code for every file. Your job is to create the files exactly as specified, run the commands, and verify each phase works.

---

## EXECUTION RULES (NON-NEGOTIABLE)

1. **Follow the bible files EXACTLY.** Do not improvise, do not change function signatures, do not modify imports, do not skip files.
2. **Never use placeholder code.** Every function must be complete and working.
3. **Never swallow exceptions.** Always log errors properly.
4. **Verify after each phase.** If a verification step fails, FIX IT before moving to the next phase. Do NOT skip broken phases.
5. **Git commit after each phase.** Use the exact commit messages specified below.
6. **Windows/PowerShell environment.** Use PowerShell syntax for all commands. Python f-strings with dict access break in PowerShell — write to .py files instead of using `python -c`.
7. **UI must be dark theme glassmorphism.** No plain/default HTML anywhere.

---

## GIT SETUP (DO THIS AFTER READING BUT BEFORE CODING)

```powershell
cd "E:/rishabh/sih/sih-2026-problem-statements-main (1)/sih-2026-problem-statements-main/depthwizard"
git init
```

Create `.gitignore`:
```
__pycache__/
*.pyc
.env
node_modules/
dist/
*.egg-info/
backend/weights/*.pth
backend/logs/*.jsonl
backend/exports/
.vite/
*.npy
.DS_Store
```

```powershell
git add .gitignore
git commit -m "chore: init repo with gitignore"
```

---

## PHASE 1: Backend Core (Tasks 0–11 from Bible)

### What to build:
- Directory structure
- `requirements.txt` + `pip install`
- `backend/app/config.py`
- `backend/app/logging_config.py`
- `backend/app/services/depth_estimator.py`
- `backend/app/services/geospatial.py`
- `backend/app/services/calibration.py`
- `backend/app/services/mesh_builder.py`
- `backend/app/services/validation.py`
- `backend/app/api/schemas.py`
- `backend/app/api/routes.py`
- `backend/app/main.py`
- All `__init__.py` files

### Subagent strategy:
If you can use subagents, split this into TWO parallel agents:
- **Agent A (Backend Services):** Create config, logging, depth_estimator, geospatial, calibration, mesh_builder, validation
- **Agent B (Backend API):** Create schemas, routes, main.py (Agent B depends on Agent A's service files existing, but can write the import-based code simultaneously)

### VERIFICATION HOOK (MUST PASS):
```powershell
cd "E:/rishabh/sih/sih-2026-problem-statements-main (1)/sih-2026-problem-statements-main/depthwizard/backend"

# Test 1: CUDA availability
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE'; print('CUDA OK:', torch.cuda.get_device_name(0))"

# Test 2: Start server (run in background, test health, then stop)
Start-Process -NoNewWindow python -ArgumentList "-m","uvicorn","app.main:app","--host","0.0.0.0","--port","8000"
Start-Sleep -Seconds 15
$response = Invoke-RestMethod -Uri "http://localhost:8000/api/health" -Method Get
Write-Host "Health check:" $response.status $response.gpu
# Kill the server after test
Stop-Process -Name "python" -Force -ErrorAction SilentlyContinue
```

**Expected:** `Health check: ok NVIDIA GeForce RTX 4060`

If it fails → fix the error → re-verify → do NOT proceed until this passes.

### GIT COMMIT:
```powershell
cd "E:/rishabh/sih/sih-2026-problem-statements-main (1)/sih-2026-problem-statements-main/depthwizard"
git add backend/
git commit -m "feat(backend): core pipeline - depth estimation, calibration, mesh builder, API routes"
```

---

## PHASE 2: Training (Tasks 12–14 from Bible)

### What to build:
- `backend/training/losses.py` (SILog + GradientMatching)
- `backend/training/dataset_gamus.py` (GAMUS dataset adapter, max 200 samples)
- `backend/training/train_gamus.py` (10 epochs, fp16, RTX 4060 optimized)

### Subagent strategy:
This is a single sequential task. One agent creates the files, then runs training.

### RUN TRAINING:
```powershell
cd "E:/rishabh/sih/sih-2026-problem-statements-main (1)/sih-2026-problem-statements-main/depthwizard/backend"
python training/train_gamus.py
```

### VERIFICATION HOOK (MUST PASS):
```powershell
# Check that best_model.pth was created
Test-Path "backend/weights/best_model.pth"
# Expected: True

# Check that training log exists and has entries
Get-Content "backend/logs/training_mvp.jsonl" -Tail 3
# Expected: JSON lines with decreasing loss values
```

**Expected:** Training completes in ~2-5 minutes. `best_model.pth` exists. Loss at epoch 10 < loss at epoch 1.

### GIT COMMIT:
```powershell
git add backend/training/
git commit -m "feat(training): GAMUS fine-tuning with SILog+GradMatch loss, 200 samples, 10 epochs"
```

---

## PHASE 3: Frontend (Tasks 15–26 from Frontend Bible)

### What to build:
- Vite + React + TypeScript scaffold
- npm dependencies: `three @react-three/fiber @react-three/drei recharts lucide-react tailwindcss @tailwindcss/vite d3-contour d3-geo`
- `src/main.tsx`
- `src/styles/globals.css` (dark glassmorphism)
- `src/utils/colormap.ts`
- `src/hooks/useInference.ts`
- `src/components/Dropzone.tsx`
- `src/components/DualViewer.tsx`
- `src/components/TerrainCanvas.tsx` (Three.js 3D flythrough)
- `src/components/FloodSimulator.tsx`
- `src/components/CrossSection.tsx`
- `src/components/ContourOverlay.tsx`
- `src/components/ColorBar.tsx`
- `src/components/SettingsPanel.tsx`
- `src/App.tsx`
- `vite.config.ts` (with proxy to backend)
- `index.html` (with Inter font)

### Subagent strategy (2 parallel agents):
- **Agent C (Frontend Core):** Create scaffold, globals.css, colormap.ts, useInference.ts, Dropzone.tsx, DualViewer.tsx, TerrainCanvas.tsx, App.tsx, vite.config.ts, index.html
- **Agent D (Frontend UVPs):** Create FloodSimulator.tsx, CrossSection.tsx, ContourOverlay.tsx, ColorBar.tsx, SettingsPanel.tsx

Agent D's components are imported by App.tsx (Agent C), so Agent C should write the import statements referencing these files — they'll exist by the time both agents finish.

### VERIFICATION HOOK (MUST PASS):

First, start the backend:
```powershell
# Terminal 1
cd "E:/rishabh/sih/sih-2026-problem-statements-main (1)/sih-2026-problem-statements-main/depthwizard/backend"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Then start frontend:
```powershell
# Terminal 2
cd "E:/rishabh/sih/sih-2026-problem-statements-main (1)/sih-2026-problem-statements-main/depthwizard/frontend"
npm run dev
```

Then verify:
```powershell
# Terminal 3 - TypeScript check
cd "E:/rishabh/sih/sih-2026-problem-statements-main (1)/sih-2026-problem-statements-main/depthwizard/frontend"
npx tsc --noEmit
# Expected: No errors (some warnings are OK)
```

**Manual verification (open browser to http://localhost:5173):**
- [ ] Dark glassmorphic UI loads (not plain white/default)
- [ ] DepthWizard header with logo visible
- [ ] Dropzone area with drag-and-drop styling
- [ ] Drop an image → processing animation → results load
- [ ] DualViewer shows RGB and colored DSM side by side
- [ ] 3D terrain renders with displacement
- [ ] WASD + mouse flythrough works
- [ ] Flood slider appears and turns terrain blue
- [ ] Cross-section button works and shows chart
- [ ] Settings panel has vertical scale slider
- [ ] Export GeoTIFF button downloads a .tif file

### GIT COMMIT:
```powershell
git add frontend/
git commit -m "feat(frontend): Three.js 3D flythrough, flood sim, cross-section, contour lines, glassmorphism UI"
```

---

## PHASE 4: Integration & Ship (Tasks 27–30 from Bible)

### What to build:
- `run_local.bat`
- `README.md`
- Final integration testing

### VERIFICATION HOOK (FINAL — ALL MUST PASS):

```powershell
# Start everything via run_local.bat
cd "E:/rishabh/sih/sih-2026-problem-statements-main (1)/sih-2026-problem-statements-main/depthwizard"
.\run_local.bat
```

Then verify ALL of these:
1. ✅ Backend health: `http://localhost:8000/api/health` returns OK
2. ✅ Frontend loads at `http://localhost:5173`
3. ✅ Upload a PNG/JPG → depth estimation completes
4. ✅ 2D DualViewer shows RGB vs DSM
5. ✅ 3D terrain renders with proper displacement
6. ✅ WASD flythrough works smoothly (≥30 FPS)
7. ✅ Flood simulator slider works in real-time
8. ✅ Cross-section shows elevation profile chart
9. ✅ Settings: vertical exaggeration slider changes terrain height
10. ✅ Export button downloads a GeoTIFF file
11. ✅ "New Image" button resets to dropzone

### FINAL GIT COMMIT:
```powershell
git add .
git commit -m "feat: DepthWizard MVP complete - depth estimation, 3D flythrough, flood sim, cross-section, GeoTIFF export"
```

---

## ERROR RECOVERY RULES

If something breaks during any phase:

1. **Read the full error message.** Do not guess.
2. **Check the bible files** for the exact code — you may have deviated.
3. **Common issues on Windows:**
   - `ModuleNotFoundError` → wrong working directory or missing `__init__.py`
   - CORS errors → check that FastAPI CORS middleware is configured with `allow_origins=["*"]`
   - Three.js black screen → check that textures are loading from base64 correctly
   - `torch.cuda.OutOfMemoryError` → reduce batch size or close other GPU apps
   - PowerShell encoding issues → use UTF-8: `[Console]::OutputEncoding = [Text.Encoding]::UTF8`
4. **Fix the root cause.** Do NOT:
   - Comment out broken code
   - Return dummy data
   - Swallow exceptions with empty `except: pass`
   - Skip verification steps

---

## SUBAGENT ALLOCATION SUMMARY

```
PHASE 1 (Backend):
  ├── Agent A: Services (depth_estimator, geospatial, calibration, mesh_builder, validation)
  └── Agent B: API layer (schemas, routes, main.py) — can write in parallel, same directory
  → MERGE → Verify backend /health → Git commit

PHASE 2 (Training):
  └── Single agent: losses.py, dataset.py, train.py → Run training → Git commit

PHASE 3 (Frontend):
  ├── Agent C: Core (scaffold, App.tsx, TerrainCanvas.tsx, Dropzone, DualViewer, hooks)
  └── Agent D: UVPs (FloodSimulator, CrossSection, ContourOverlay, ColorBar, SettingsPanel)
  → MERGE → Verify frontend builds → Git commit

PHASE 4 (Integration):
  └── Single agent: run_local.bat, README, final test → Git commit
```

---

## WHAT SUCCESS LOOKS LIKE

When you're done, the user should be able to:

1. Run `run_local.bat`
2. Open `http://localhost:5173`
3. Drop a satellite image
4. See the AI estimate the terrain height
5. Fly through the 3D terrain with WASD + mouse
6. Drag the flood slider to simulate flooding
7. Click cross-section to see an elevation profile
8. Export the result as a GeoTIFF

**This is a hackathon project for ISRO. It must look polished, work reliably, and demonstrate real AI capability — not a toy demo.**

---

## NOW START. Read the two bible files, then execute Phase 1.
