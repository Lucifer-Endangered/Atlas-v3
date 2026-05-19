@echo off
REM ============================================================
REM ATLAS V2 — One-Click Setup for Windows
REM ============================================================
REM Run this on the target laptop with Autodesk Inventor installed.
REM Prerequisites: Python 3.10+ must be installed.
REM ============================================================

echo ============================================================
echo   ATLAS V2 — Setting up ML Assembly Automation
echo ============================================================

REM Create project directory
set ATLAS_DIR=C:\ATLAS
if not exist "%ATLAS_DIR%" mkdir "%ATLAS_DIR%"
cd /d "%ATLAS_DIR%"

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

echo.
echo [1/4] Creating virtual environment...
if not exist "venv" python -m venv venv
call venv\Scripts\activate.bat

echo.
echo [2/4] Installing dependencies...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy scikit-learn requests

echo.
echo [3/4] Verifying model files...
if not exist "v2_checkpoints\atlas_v2_best.pt" (
    echo ERROR: Model checkpoint not found!
    echo Copy the following files to %ATLAS_DIR%\v2_checkpoints\
    echo   - atlas_v2_best.pt
    echo   - norm_mean.npy
    echo   - norm_std.npy
    echo.
    echo Also copy these Python files to %ATLAS_DIR%\
    echo   - atlas_v2_server.py
    echo   - atlas_v2_train.py
    echo   - atlas_v2_data.py
    pause
    exit /b 1
)

echo   Model found: v2_checkpoints\atlas_v2_best.pt
echo   Normalization: v2_checkpoints\norm_mean.npy, norm_std.npy

echo.
echo [4/4] Testing server startup...
python atlas_v2_server.py --model-dir ./v2_checkpoints --port 5050 --confidence 0.65

pause
