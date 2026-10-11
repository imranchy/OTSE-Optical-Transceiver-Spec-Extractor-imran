@echo off
REM Quick Start Script for GNPy Simulator

echo ========================================
echo   GNPy Optical Network Simulator
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo [1/3] Checking Python installation...
python --version

echo.
echo [2/3] Installing/Updating dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [3/3] Starting GNPy Simulator...
echo.
echo The application will open in your browser at http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

REM Activate virtual environment if it exists
if exist .venv\Scripts\activate (
    call .venv\Scripts\activate
    streamlit run app.py --server.port 8501
) else (
    streamlit run app.py
)

pause
