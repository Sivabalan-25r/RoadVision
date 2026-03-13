# RoadVision - Quick Start

Get RoadVision running in 3 minutes.

## Prerequisites

- Python 3.8+ installed
- Internet connection (for first-time setup)

## Steps

### 1. Install Dependencies (First Time Only)

```bash
cd RoadVision/backend-python
pip install -r requirements.txt
```

⏱️ Takes ~5 minutes

### 2. Start the Application

**Windows:**
```bash
cd RoadVision
start.bat
```

**Linux/Mac:**
```bash
cd RoadVision
chmod +x start.sh
./start.sh
```

### 3. Open Your Browser

Frontend: **http://localhost:3000**

API Docs: **http://localhost:8000/docs**

## Usage

1. Click "Choose Video" or drag & drop
2. Select a traffic video (MP4, MOV, WEBM)
3. Click "Analyze Video"
4. View detection results

## Troubleshooting

**Backend won't start?**
- Check if `models/license_plate_detector.pt` exists
- Run: `pip install -r requirements.txt`

**Can't connect to backend?**
- Verify backend is running at http://localhost:8000/health
- Check firewall settings

**No detections found?**
- Ensure video shows clear license plates
- Check video quality (720p+ recommended)

## Need Help?

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for detailed instructions.

---

**That's it! You're ready to detect illegal plates.** 🚗
