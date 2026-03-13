@echo off
REM RoadVision Startup Script (Windows)

echo ==========================================
echo   RoadVision - Starting Application
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    echo Please install Python 3.8 or higher
    pause
    exit /b 1
)

echo Python found
echo.

REM Navigate to backend directory
cd backend-python

REM Check if virtual environment exists
if not exist "venv" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    echo Virtual environment created
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if dependencies are installed
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Dependencies not installed. Installing...
    pip install -r requirements.txt
    echo Dependencies installed
) else (
    echo Dependencies already installed
)

REM Check for model files
echo.
echo Checking model files...
if not exist "models\license_plate_detector.pt" (
    echo Error: license_plate_detector.pt not found in models/
    echo Please download the model and place it in backend-python/models/
    pause
    exit /b 1
)
echo Model files found

echo.
echo ==========================================
echo   Starting Backend Server
echo ==========================================
echo Backend will run at: http://localhost:8000
echo API docs available at: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.

REM Start the backend server
uvicorn main:app --reload --port 8000
