@echo off
setlocal EnableExtensions

cd /d D:\Depth-Wizard

REM ============================================================
REM LOG FILES
REM ============================================================

set "RUNLOG=D:\Depth-Wizard\run1.txt"
set "ERRORLOG=D:\Depth-Wizard\error1.txt"

REM ============================================================
REM Redirect the entire BAT console output to run1.txt
REM while ALSO displaying it in the Task Scheduler process.
REM ============================================================

call :main >> "%RUNLOG%" 2>&1
set "FINAL_EXIT=%errorlevel%"

echo.
echo ============================================================
echo QUEUE EXITED
echo Exit code: %FINAL_EXIT%
echo Finished: %date% %time%
echo ============================================================

exit /b %FINAL_EXIT%


:main

echo.
echo ============================================================
echo DEPTH-WIZARD P1 TRAINING QUEUE
echo ============================================================
echo Started:  %date% %time%
echo Run log:  %RUNLOG%
echo Error log: %ERRORLOG%
echo ============================================================
echo.

REM ============================================================
REM STAGE 1
REM ============================================================

echo.
echo ============================================================
echo STARTING STAGE 1
echo Time: %date% %time%
echo ============================================================
echo.

D:\Depth-Wizard\.venv\Scripts\python.exe upgrade\training\train_metric.py --resume --epochs 50 --warmup 3 --enc-lr 2.5e-6 --head-lr 2.5e-5 --batch 2 --grad-accum 8 --crop 504 --workers 4

set "STAGE1_EXIT=%errorlevel%"

echo.
echo Stage 1 exit code: %STAGE1_EXIT%
echo Stage 1 finished: %date% %time%

if not "%STAGE1_EXIT%"=="0" (
    echo.
    echo ============================================================
    echo STAGE 1 FAILED
    echo Exit code: %STAGE1_EXIT%
    echo Time: %date% %time%
    echo ============================================================
    echo [STAGE 1 FAILED] Exit code %STAGE1_EXIT% - %date% %time%>>"%ERRORLOG%"
    echo See train_stage1.log and run1.txt for details.>>"%ERRORLOG%"
    exit /b %STAGE1_EXIT%
)

echo.
echo ============================================================
echo STAGE 1 COMPLETED SUCCESSFULLY
echo Starting STAGE 2
echo Time: %date% %time%
echo ============================================================
echo.


REM ============================================================
REM STAGE 2
REM ============================================================

echo.
echo ============================================================
echo STARTING STAGE 2
echo Time: %date% %time%
echo ============================================================
echo.

D:\Depth-Wizard\.venv\Scripts\python.exe upgrade\training\train_metric.py --resume --w-silog 1.0 --w-l1 1.0 --w-grad 0.5 --epochs 60 --crop 504 --workers 4

set "STAGE2_EXIT=%errorlevel%"

echo.
echo Stage 2 exit code: %STAGE2_EXIT%
echo Stage 2 finished: %date% %time%

if not "%STAGE2_EXIT%"=="0" (
    echo.
    echo ============================================================
    echo STAGE 2 FAILED
    echo Exit code: %STAGE2_EXIT%
    echo Time: %date% %time%
    echo ============================================================
    echo [STAGE 2 FAILED] Exit code %STAGE2_EXIT% - %date% %time%>>"%ERRORLOG%"
    echo See train_stage2.log and run1.txt for details.>>"%ERRORLOG%"
    exit /b %STAGE2_EXIT%
)

echo.
echo ============================================================
echo STAGE 2 COMPLETED SUCCESSFULLY
echo Starting STAGE 3
echo Time: %date% %time%
echo ============================================================
echo.


REM ============================================================
REM STAGE 3
REM ============================================================

echo.
echo ============================================================
echo STARTING STAGE 3
echo Time: %date% %time%
echo ============================================================
echo.

D:\Depth-Wizard\.venv\Scripts\python.exe upgrade\training\train_metric.py --resume --w-silog 1.0 --w-l1 1.0 --w-grad 0.5 --w-lt 0.5 --epochs 60 --crop 504 --workers 4

set "STAGE3_EXIT=%errorlevel%"

echo.
echo Stage 3 exit code: %STAGE3_EXIT%
echo Stage 3 finished: %date% %time%

if not "%STAGE3_EXIT%"=="0" (
    echo.
    echo ============================================================
    echo STAGE 3 FAILED
    echo Exit code: %STAGE3_EXIT%
    echo Time: %date% %time%
    echo ============================================================
    echo [STAGE 3 FAILED] Exit code %STAGE3_EXIT% - %date% %time%>>"%ERRORLOG%"
    echo See train_stage3.log and run1.txt for details.>>"%ERRORLOG%"
    exit /b %STAGE3_EXIT%
)


REM ============================================================
REM EVERYTHING FINISHED
REM ============================================================

echo.
echo ============================================================
echo ALL THREE STAGES FINISHED SUCCESSFULLY
echo ============================================================
echo Finished: %date% %time%
echo.
echo Logs:
echo   %RUNLOG%
echo   %ERRORLOG%
echo   upgrade\outputs\train_stage1.log
echo   upgrade\outputs\train_stage2.log
echo   upgrade\outputs\train_stage3.log
echo.
echo Checkpoints:
echo   upgrade\checkpoints\metric_best.pth
echo   upgrade\checkpoints\metric_last.pth
echo ============================================================
echo.

exit /b 0
