@echo off
REM EvasionEye Startup Script (Windows)

echo ==========================================
echo   EvasionEye - Starting Application
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    exit /b 1
)

echo Python found
echo.

REM Navigate to backend directory
cd backend-python

echo Activating virtual environment...

if not exist ..\venv310\Scripts\activate.bat goto create_venv

call ..\venv310\Scripts\activate.bat
python --version >nul 2>&1
if errorlevel 1 goto repair_venv
goto sync_deps

:repair_venv
echo Virtual environment appears broken. Re-creating...
cd ..
rmdir /s /q venv310
cd backend-python

:create_venv
echo Creating virtual environment (3.10)...
py -3.10 -m venv ..\venv310
call ..\venv310\Scripts\activate.bat

:sync_deps
echo Synchronizing dependencies...
pip install -r requirements.txt
echo Dependencies synchronized

echo.
echo Checking model files...
if not exist "models\yolov26-license-plate.pt" (
    echo Error: yolov26-license-plate.pt not found in models/
    echo Please place the 'yolov26-license-plate.pt' file in the 'backend-python/models/' directory before starting the application.
    exit /b 1
)
echo Model files found

echo.
echo ==========================================
echo   Starting Backend Server
echo ==========================================
echo Backend will run at: http://localhost:8000
echo.

REM Start the backend server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --lifespan on --timeout-graceful-shutdown 5
