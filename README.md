# Drishya — Video-based Detection Dashboard

Simple, local video & webcam monitoring with detectors for anomalies, pose/behavior, QR attendance, and anticheat.

Problem → Solution
--------------------
Problem: Deploying and testing multiple video detectors (anomaly, pose, QR, anticheat) across local cameras and files is tedious and fragmented.

Solution: Drishya provides a lightweight frontend (web UI) and a Python backend that run detectors, publish camera streams, and surface alerts/attendance in one integrated workflow.

Quick Start
-----------
Prerequisites
- Python 3.8+ (backend)
- Node 16+/npm or Yarn (frontend)

Run the backend

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python deps and start services:

```bash
pip install -r backend/requirements.txt
python backend/start_services.py
# or to run the server directly:
python backend/python_server.py
```

Run the frontend

```bash
cd frontend
npm install
npm run dev
# build for production:
npm run build
```

Open the UI at the address shown by Vite (typically http://localhost:5173) and ensure the backend is reachable (check backend logs for port).

How It Works (high-level)
-------------------------
- Frontend: Vite + React app that displays camera feeds, detector outputs, and alerts.
- Backend: Python services that capture/process video, run detector modules, and serve results via WebSocket/HTTP.
- Components:
  - Camera publisher(s) capture streams (`backend/camera_publisher.py`).
  - Detector modules process frames (`backend/anomaly_detector.py`, `backend/pose_behavior_detector.py`, `backend/anticheat_detector.py`, `backend/qr_detector.py`).
  - `backend/python_server.py` ties detectors to a socket/HTTP interface consumed by the frontend.

Key Features
------------
- Multiple detectors: anomaly, pose/behavior, anticheat, QR attendance.
- Local-first: runs on local hardware with minimal external dependencies.
- Batch and live: supports file test videos and live webcam feeds.
- Simple UI: grid view of camera feeds and quick alert/attendance panels.

Configuration / Usage
---------------------
- Environment and ports: check `backend/python_server.py` for server port and settings.
- Detector settings and zones: see `backend/anticheat1/zones.json` and detector modules for tuning parameters.
- QR attendance: QR data and snapshot outputs live under `backend/qr/` and `backend/qr_snapshots/`.

Tips
- If a detector requires a native dependency (OpenCV, media libs), install OS packages first (e.g., `sudo apt install libsm6 libxrender1 libxext6` on Debian/Ubuntu).
- Use `backend/start_services.py` to start all backend pieces together; use `backend/stop_services.py` to stop them.

Where to look next
- Frontend components: `frontend/src/components/` for UI code.
- Backend detectors: `backend/` Python files.

Contributing
------------
PRs welcome — follow the existing code style and add tests where useful.