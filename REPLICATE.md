# Drishya — Replication Guide

This document lists every technology, dependency, and step required to replicate Drishya locally. Follow the numbered sections in order to clone, install, configure, and run the full stack (backend detectors + frontend UI). This guide assumes Linux (tested on Ubuntu/Debian-like systems).

1) Technology Summary
---------------------
- Frontend
  - Vite (dev server, build)
  - React (TypeScript)
  - Tailwind CSS
  - shadcn/ui, Lucide React, Framer Motion
- Backend
  - Python (3.8+ recommended)
  - Flask + flask-cors (simple stream API in `backend/python_server.py`)
  - Ultralytics (YOLOv8) for detection (`ultralytics`)
  - OpenCV (`opencv-python`) for captured frames
  - PyTorch + torchvision (used by `ultralytics` and models)
  - Redis (IPC/stream coordination used by startup scripts)
  - deep-sort-realtime (tracking)
  - pyzbar, Pillow, numpy, pandas
  - CLIP (git dependency listed in `backend/requirements.txt`)
- Node stack (backend helper): Express, dotenv, cors (see `backend/package.json`)

2) Repo layout (important files to read)
---------------------------------------
- `backend/` — Python detector services, service scripts, and `requirements.txt`
- `backend/start_services.py` — orchestrates starting Redis and all detector services (uses `conda run -n gen` by default)
- `backend/stop_services.py` — stops detector processes and cleans PID files
- `backend/python_server.py` — simple Flask-based video streaming server (YOLO demo)
- `backend/src/index.js` and `backend/package.json` — small Node/Express helper API
- `frontend/` — Vite + React UI. See `frontend/package.json` and `frontend/src/components/`
- `backend/anticheat1/zones.json` — example detector configuration
- `backend/qr/`, `backend/qr_snapshots/` — QR outputs and data

3) Prerequisites (system-level)
--------------------------------
- Git (clone the repo)
- Node.js (16+) and npm or Yarn
- Python 3.8+
- Conda (optional but used by scripts) or a plain Python venv
- Redis server
- OS packages for OpenCV (Debian/Ubuntu example):

```bash
sudo apt update
sudo apt install -y build-essential libsm6 libxrender1 libxext6 redis-server
```

4) Recommended environment choices
----------------------------------
- Option A — Conda (matches existing scripts): create an environment named `gen` used by `start_services.py`:

```bash
conda create -n gen python=3.10 -y
conda activate gen
pip install -r backend/requirements.txt
```

- Option B — Python venv (alternative):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

Notes: `backend/start_services.py` and several scripts call `conda run -n gen ...`. If you prefer virtualenv, either run scripts manually (explained below) or edit `start_services.py` to remove `conda` invocations.

5) Step-by-step Setup and Run (full replication)
-----------------------------------------------
Clone the repo:

```bash
git clone <repo-url> Drishya
cd Drishya
```

a) Install and start Redis (system service):

```bash
# Debian/Ubuntu
sudo apt install -y redis-server
sudo systemctl enable --now redis-server
redis-cli ping  # should return PONG
```

b) Backend Python environment & deps:

```bash
# Using conda (recommended for parity with scripts)
conda create -n gen python=3.10 -y
conda activate gen
pip install -r backend/requirements.txt

# If any packages fail (e.g., torch), follow their platform-specific install guides
```

c) Frontend setup:

```bash
cd frontend
npm install
cd ..
```

d) Prepare environment file for Python server (optional)
- `backend/python_server.py` loads `.env.python`. Create it when you want to override `PORT` or other settings:

Example `.env.python` at `backend/.env.python`:

```
PORT=5001
# any other env vars you want
```

e) Start everything (two options):

- Option 1 — Use orchestration scripts (starts Redis if missing, writes logs):

```bash
# Run from repository root (these scripts call `conda run -n gen` internally)
conda run -n gen python backend/start_services.py
```

This script will:
- verify Redis is available (attempt to `systemctl start` if not)
- start `camera_publisher.py`, `anticheat_detector.py`, `qr_detector.py`, and `anomaly_detector.py` under `backend/` (logs stored under `./logs`)

- Option 2 — Manual start (recommended when debugging):

```bash
# Activate environment you used to install deps (conda or venv)
source activate gen  # or: conda activate gen
python backend/python_server.py &
python backend/camera_publisher.py &
python backend/anticheat_detector.py &
python backend/qr_detector.py &
python backend/anomaly_detector.py &
```

f) Start frontend dev server:

```bash
cd frontend
npm run dev
# open http://localhost:5173 (Vite default)
```

6) Ports and endpoints
----------------------
- Frontend (Vite dev): http://localhost:5173
- Backend Node helper server: default PORT in `backend/src/index.js` (env aware, often 5000)
- Python detector endpoints (from `start_services.py` printout):
  - Anomaly / video feed: `http://localhost:5001/api/video_feed`
  - Anti-cheat: `http://localhost:5002/` (endpoints under `/api/`)
  - QR detector: `http://localhost:5003/`
  - Camera publisher: `http://localhost:5004/`

7) Detector workflows (how each part works)
-------------------------------------------
- Camera publisher (`backend/camera_publisher.py`)
  - Captures camera frames (webcam or video files) and exposes a raw feed endpoint (used by frontend or other detector services).
  - Works as the raw source for downstream detectors.
- Anomaly detector (`backend/anomaly_detector.py`)
  - Typically consumes camera frames and runs an object/motion detection model (YOLO or custom) to find anomalous events and generate stats/alerts.
- QR detector (`backend/qr_detector.py`)
  - Reads frames, decodes QR codes (pyzbar), writes attendance/alerts into `backend/qr/` and stores snapshots in `backend/qr_snapshots/`.
- Anti-cheat detector (`backend/anticheat_detector.py` / `backend/anticheat1/`)
  - Uses zone definitions in `backend/anticheat1/zones.json` and logic in `backend/anticheat1/main_optimized.py` to flag cheating behaviors in defined zones. Outputs to `backend/anticheat1/alerts.csv`.

8) Data, logs and outputs
-------------------------
- Logs: `./logs/` (created by `start_services.py`)
- QR: `backend/qr/attendance.csv` and `backend/qr/alerts.csv` (runtime outputs)
- QR snapshots: `backend/qr_snapshots/`
- Anticheat example alerts: `backend/anticheat1/alerts.csv`

9) Verification steps
---------------------
After starting services:

```bash
# Check Redis
redis-cli ping

# Check Python server health
curl http://localhost:5001/api/health

# Check Node backend
curl http://localhost:5000/api/health

# Open the frontend at http://localhost:5173 and confirm camera grid loads
```

10) Common troubleshooting
-------------------------
- Redis not running: start with `sudo systemctl start redis-server` and re-run `start_services.py`.
- OpenCV camera access fails:
  - Confirm no other process is using the camera.
  - Run `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"` inside the activated env.
- Torch/Ultralytics install errors: follow platform-specific instructions from PyTorch (`https://pytorch.org/get-started/locally/`) and then re-run `pip install -r backend/requirements.txt`.
- Permission or Conda issues from `start_services.py`: run the scripts manually in your activated environment to see stdout instead of background logs.

11) Reproducing without Conda (if you prefer venv)
------------------------------------------------
- Create and activate a venv, install `requirements.txt`, run each Python service directly. Replace `conda run -n gen python <script>` with `python <script>` from the activated venv.

12) Notes about models and data
------------------------------
- YOLO / ultralytics: `backend/python_server.py` expects a YOLO model (`yolov8n.pt`) or will attempt to load the default. For better accuracy train or supply a custom weights file and update the script path.
- CLIP: listed as a git dependency; ensure `git` and build toolchain are installed before running pip install.

13) Security and production considerations
-----------------------------------------
- This repository is intended for local development, testing and small deployments. For production:
  - Add authentication, TLS, and proper CORS rules.
  - Move long-running processes to a process manager (systemd, supervisor, or Docker-compose).
  - Containerize services and pin package versions in a lockfile.

14) Quick checklist to hand to your friend
-----------------------------------------
- Clone repo
- Install Redis
- Create Python environment (conda `gen` or venv)
- Install Python deps (`pip install -r backend/requirements.txt`)
- Install frontend deps (`cd frontend && npm install`)
- Create `.env.python` in `backend/` only if you need to change PORT
- Start services with `conda run -n gen python backend/start_services.py` or start individual scripts manually
- Start frontend with `cd frontend && npm run dev`

15) Where to look in code
-------------------------
- Frontend UI components: `frontend/src/components/` — dashboard, `CameraGrid.tsx`, `WebcamFeed.tsx`
- Backend entry & helpers: `backend/python_server.py`, `backend/src/index.js`, `backend/start_services.py`, `backend/stop_services.py`
- Detector modules: `backend/anomaly_detector.py`, `backend/qr_detector.py`, `backend/anticheat_detector.py`, `backend/pose_behavior_detector.py`

If you want, I can now:
- Expand any specific detector's replication steps (e.g., how to train or supply YOLO weights), or
- Create a small Docker Compose to fully reproduce the stack. Which would you like next?
