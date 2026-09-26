@echo off
setlocal enabledelayedexpansion

title Depth-Wizard - Open-Canopy Dataset Downloader

cd /d "%~dp0"

echo ============================================================
echo DEPTH-WIZARD: OPEN-CANOPY 2023 RESILIENT DOWNLOADER
echo ============================================================
echo Destination: %~dp0Open-Canopy
echo Mode:        Pair-by-pair (LiDAR + SPOT) with instant linking
echo Resume:      Supported (Ctrl+C safe, rerun anytime)
echo ============================================================
echo.

set "PYTHON_EXE="
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
    where python >nul 2>nul
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=python"
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python environment not found!
    echo Please make sure .venv exists or Python is installed.
    pause
    exit /b 1
)

set PYTHONUNBUFFERED=1

"%PYTHON_EXE%" "%~dp0download_all_aux_datasets.py" --dataset opencanopy %*

if %errorlevel% neq 0 (
    echo.
    echo ============================================================
    echo Download paused or exited with code %errorlevel%.
    echo You can rerun this script anytime to resume.
    echo ============================================================
) else (
    echo.
    echo ============================================================
    echo OPEN-CANOPY DOWNLOAD COMPLETED SUCCESSFULLY!
    echo ============================================================
)

pause
