---
title: Traffic Signal Optimization AI Agent
emoji: 🚦
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.30.0
app_file: dashboard/app.py
pinned: false
---

# Traffic Signal Optimization on Agent

An AI-powered traffic management system that uses computer vision and adaptive signal control to optimize traffic flow at intersections.

## Features

| Module | Description |
|--------|-------------|
| **Vehicle Detection & Count** | YOLOv8 + ByteTrack for real-time vehicle detection and accurate counting |
| **Adaptive Traffic Signal** | AI agent that dynamically adjusts signal timing based on traffic density |
| **Night Mode** | CLAHE-based low-light enhancement for night-time detection |
| **Helmet Detection** | Fine-tuned YOLOv8 to detect helmet violations on motorcycles |
| **Accident Detection** | AI-based accident/collision detection with real-time alerts |
| **Analytics Dashboard** | Streamlit-based web dashboard with interactive graphs and reports |
| **Database Logging** | Automatic MySQL/SQLite storage with time-based query filters |

## Tech Stack

- **Language:** Python 3.10+
- **AI Models:** YOLOv8 (Ultralytics)
- **Tracking:** ByteTrack
- **Video:** OpenCV
- **Database:** MySQL / SQLite
- **Dashboard:** Streamlit + Plotly
- **GPU:** NVIDIA RTX 3050 (CUDA accelerated)
- **Containerization:** Docker

## Project Structure

```
AI project/
|-- config/             # Central configuration
|-- core/
|   |-- detection/      # Vehicle, Helmet, Accident detectors
|   |-- tracking/       # ByteTrack vehicle tracker
|   |-- agent/          # Adaptive signal controller
|-- database/           # DB models and manager
|-- dashboard/          # Streamlit web dashboard
|-- utils/              # Night enhancement, video helpers
|-- assets/             # Videos and images
|-- data/models/        # YOLO model weights
|-- tests/              # Unit tests
|-- main.py             # Main pipeline entry point
|-- requirements.txt    # Python dependencies
|-- Dockerfile          # Docker containerization
|-- README.md           # Documentation & Hugging Face metadata
```

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Web Dashboard
streamlit run dashboard/app.py

# 5. Run Video Pipeline (CLI)
python main.py --video assets/videos/sample_traffic.mp4

# 6. Run Live Webcam
python main.py --webcam
```
