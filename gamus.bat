@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM DEPTH-WIZARD GAMUS TRAINING QUEUE (STAGES 1 - 3)
REM ============================================================
REM Runs all three loss stages on 100% of the local GAMUS dataset:
REM   Stage 1: Metric Baseline (50 epochs)
REM   Stage 2: Edge-Aware Loss (60 epochs, resumes from Stage 1)
REM   Stage 3: Long-Tail + Edge Loss (70 epochs, resumes from Stage 2)
REM ============================================================

cd /d "D:\Depth-Wizard"

set "ROOT=D:\Depth-Wizard"

REM Automatically resolve to project .venv python
if exist "%ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
) else (
    set "PYTHON=python"
)

set "TRAIN=%ROOT%\upgrade\training\train_metric.py"

set "RUNLOG=%ROOT%\run1.txt"
set "ERRORLOG=%ROOT%\error1.txt"

set "STAGE1_LOG=%ROOT%\upgrade\outputs\train_stage1.log"
set "STAGE2_LOG=%ROOT%\upgrade\outputs\train_stage2.log"
set "STAGE3_LOG=%ROOT%\upgrade\outputs\train_stage3.log"

REM Force 100% offline mode for Hugging Face Hub / Transformers
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"

REM Local GAMUS dataset path
set "GAMUS_ROOT=%ROOT%\GAMUS"

REM ============================================================
REM VERIFY ACTIVE VIRTUAL ENVIRONMENT
REM ============================================================

echo.
echo ============================================================
echo VERIFYING ACTIVE VIRTUAL ENVIRONMENT
echo ============================================================
echo.

"%PYTHON%" --version >nul 2>&1

if errorlevel 1 (
    echo ERROR: Python is not available at: %PYTHON%
    echo.
    echo Please make sure .venv is installed:
    echo ..venv\Scripts\Activate.ps1
    echo.
    exit /b 3
)

echo Python executable:
"%PYTHON%" -c "import sys; print(sys.executable)"

echo.
echo Python version:
"%PYTHON%" --version

REM ============================================================
REM VERIFY TRAINING SCRIPT & DIRECTORIES
REM ============================================================

if not exist "%TRAIN%" (
    echo ERROR: Training script not found:
    echo %TRAIN%
    exit /b 3
)

if not exist "%GAMUS_ROOT%" (
    echo ERROR: GAMUS root directory not found at:
    echo %GAMUS_ROOT%
    exit /b 3
)

if not exist "%ROOT%\upgrade\outputs" mkdir "%ROOT%\upgrade\outputs"
if not exist "%ROOT%\upgrade\logs" mkdir "%ROOT%\upgrade\logs"
if not exist "%ROOT%\upgrade\checkpoints" mkdir "%ROOT%\upgrade\checkpoints"
if not exist "%ROOT%\upgrade\data\gamus_cache" mkdir "%ROOT%\upgrade\data\gamus_cache"

REM ============================================================
REM PYTHON / CUDA VERIFICATION
REM ============================================================

echo.
echo ============================================================
echo PYTHON / CUDA VERIFICATION
echo ============================================================
echo.

echo PyTorch:
"%PYTHON%" -c "import torch; print(torch.__version__)"

echo.
echo CUDA available:
"%PYTHON%" -c "import torch; print(torch.cuda.is_available())"

echo.
echo GPU:
"%PYTHON%" -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU')"

echo.
echo ============================================================
echo PYTHON VERIFICATION COMPLETE
echo ============================================================
echo.

REM ============================================================
REM START MASTER QUEUE
REM ============================================================

echo.
echo ============================================================
echo LAUNCHING GAMUS TRAINING QUEUE (Stages 1 - 3)
echo Standard output redirected to: %RUNLOG%
echo Error output redirected to:    %ERRORLOG%
echo.
echo To monitor live progress in PowerShell, run:
echo   Get-Content -Path "%RUNLOG%" -Wait -Tail 30
echo ============================================================
echo.

call :main >> "%RUNLOG%" 2>&1
set "FINAL_EXIT=%errorlevel%"

echo.
echo ============================================================
echo GAMUS QUEUE FINISHED
echo Exit code: %FINAL_EXIT%
echo Finished: %date% %time%
echo ============================================================

exit /b %FINAL_EXIT%

:main

echo.
echo ============================================================
echo DEPTH-WIZARD GAMUS TRAINING QUEUE (OFFLINE MODE)
echo ============================================================
echo Started: %date% %time%
echo Working directory: %ROOT%
echo Python: %PYTHON%
echo Training script: %TRAIN%
echo Hugging Face: OFFLINE (Local Datasets Only)
echo GAMUS root: %GAMUS_ROOT%
echo DataLoader workers: 0
echo ============================================================
echo.

echo Python version:
"%PYTHON%" --version

echo.
echo Python executable:
"%PYTHON%" -c "import sys; print(sys.executable)"

REM ============================================================
REM STAGE 1: GAMUS METRIC BASELINE (50 TOTAL EPOCHS)
REM ============================================================

echo.
echo ============================================================
echo STAGE 1 START: GAMUS METRIC BASELINE
echo Time: %date% %time%
echo Target: 50 total epochs
echo Dataset: GAMUS (Offline, %GAMUS_ROOT%)
echo DataLoader workers: 0
echo ============================================================
echo.

echo [STAGE 1] Starting Python process... >> "%ERRORLOG%"

"%PYTHON%" "%TRAIN%" --offline --gamus-root "%GAMUS_ROOT%" --resume --epochs 50 --warmup 3 --enc-lr 2.5e-6 --head-lr 2.5e-5 --batch 2 --grad-accum 8 --crop 504 --workers 0 2>> "%ERRORLOG%"

set "STAGE1_EXIT=%errorlevel%"

echo.
echo STAGE 1 exit code: %STAGE1_EXIT%
echo STAGE 1 finished: %date% %time%
echo.

if not "%STAGE1_EXIT%"=="0" (
    echo ============================================================ >> "%ERRORLOG%"
    echo STAGE 1 FAILED >> "%ERRORLOG%"
    echo Exit code: %STAGE1_EXIT% >> "%ERRORLOG%"
    echo Time: %date% %time% >> "%ERRORLOG%"
    echo ============================================================ >> "%ERRORLOG%"

    echo.
    echo ============================================================
    echo STAGE 1 FAILED - QUEUE STOPPED
    echo Exit code: %STAGE1_EXIT%
    echo ============================================================
    exit /b %STAGE1_EXIT%
)

echo ============================================================
echo STAGE 1 PROCESS COMPLETED SUCCESSFULLY
echo ============================================================
echo.

REM ============================================================
REM STAGE 2: EDGE-AWARE LOSS (60 TOTAL EPOCHS, RESUMES FROM STAGE 1)
REM ============================================================

echo.
echo ============================================================
echo STAGE 2 START: EDGE-AWARE LOSS
echo Time: %date% %time%
echo Target: 60 total epochs
echo Loss: Sobel edge-aware (w_grad=0.5)
echo Dataset: GAMUS (Offline, %GAMUS_ROOT%)
echo DataLoader workers: 0
echo ============================================================
echo.

echo [STAGE 2] Starting Python process... >> "%ERRORLOG%"

"%PYTHON%" "%TRAIN%" --offline --gamus-root "%GAMUS_ROOT%" --resume --w-silog 1.0 --w-l1 1.0 --w-grad 0.5 --epochs 60 --crop 504 --workers 0 2>> "%ERRORLOG%"

set "STAGE2_EXIT=%errorlevel%"

echo.
echo STAGE 2 exit code: %STAGE2_EXIT%
echo STAGE 2 finished: %date% %time%
echo.

if not "%STAGE2_EXIT%"=="0" (
    echo ============================================================ >> "%ERRORLOG%"
    echo STAGE 2 FAILED >> "%ERRORLOG%"
    echo Exit code: %STAGE2_EXIT% >> "%ERRORLOG%"
    echo Time: %date% %time% >> "%ERRORLOG%"
    echo ============================================================ >> "%ERRORLOG%"

    echo.
    echo ============================================================
    echo STAGE 2 FAILED - QUEUE STOPPED
    echo Exit code: %STAGE2_EXIT%
    echo ============================================================
    exit /b %STAGE2_EXIT%
)

echo ============================================================
echo STAGE 2 PROCESS COMPLETED SUCCESSFULLY
echo ============================================================
echo.

REM ============================================================
REM STAGE 3: LONG-TAIL + EDGE-AWARE LOSS (70 TOTAL EPOCHS, RESUMES FROM STAGE 2)
REM ============================================================

echo.
echo ============================================================
echo STAGE 3 START: LONG-TAIL + EDGE-AWARE LOSS
echo Time: %date% %time%
echo Target: 70 total epochs
echo Loss: Sobel edge (0.5) + Long-tail (0.5)
echo Dataset: GAMUS (Offline, %GAMUS_ROOT%)
echo DataLoader workers: 0
echo ============================================================
echo.

echo [STAGE 3] Starting Python process... >> "%ERRORLOG%"

"%PYTHON%" "%TRAIN%" --offline --gamus-root "%GAMUS_ROOT%" --resume --w-silog 1.0 --w-l1 1.0 --w-grad 0.5 --w-lt 0.5 --epochs 70 --crop 504 --workers 0 2>> "%ERRORLOG%"

set "STAGE3_EXIT=%errorlevel%"

echo.
echo STAGE 3 exit code: %STAGE3_EXIT%
echo STAGE 3 finished: %date% %time%
echo.

if not "%STAGE3_EXIT%"=="0" (
    echo ============================================================ >> "%ERRORLOG%"
    echo STAGE 3 FAILED >> "%ERRORLOG%"
    echo Exit code: %STAGE3_EXIT% >> "%ERRORLOG%"
    echo Time: %date% %time% >> "%ERRORLOG%"
    echo ============================================================ >> "%ERRORLOG%"

    echo.
    echo ============================================================
    echo STAGE 3 FAILED - QUEUE STOPPED
    echo Exit code: %STAGE3_EXIT%
    echo ============================================================
    exit /b %STAGE3_EXIT%
)

REM ============================================================
REM COMPLETE
REM ============================================================

echo.
echo ============================================================
echo ALL THREE GAMUS STAGES COMPLETED SUCCESSFULLY
echo ============================================================
echo Finished: %date% %time%
echo.
echo Master log:
echo %RUNLOG%
echo.
echo Error log:
echo %ERRORLOG%
echo.
echo Checkpoints:
echo %ROOT%\upgrade\checkpoints\metric_best.pth
echo %ROOT%\upgrade\checkpoints\metric_last.pth
echo.
echo Final training target: 70 total epochs
echo DataLoader workers: 0
echo ============================================================
echo.

exit /b 0
