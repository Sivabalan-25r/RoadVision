#!/bin/bash
# EvasionEye Startup Script (Linux/Mac)

echo "=========================================="
echo "  EvasionEye - Starting Application"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if backend dependencies are installed
cd backend-python
if [ ! -d "venv" ]; then
    echo "⚠ Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Synchronizing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies synchronized"

# Check for model files
echo ""
echo "Checking model files..."
if [ ! -f "models/license_plate_detector.pt" ]; then
    echo "❌ Error: license_plate_detector.pt not found in models/"
    echo "Please download the model and place it in backend-python/models/"
    exit 1
fi
echo "✓ Model files found"

echo ""
echo "=========================================="
echo "  Starting Backend Server"
echo "=========================================="
echo "Backend will run at: http://localhost:8000"
echo "API docs available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the backend server
uvicorn main:app --reload --port 8000
