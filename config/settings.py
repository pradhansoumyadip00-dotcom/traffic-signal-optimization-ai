"""
Traffic Signal Optimization on Agent - Central Configuration
=============================================================
All project-wide settings live here so every module reads from
one single source of truth.
"""

import os
from pathlib import Path

# --- Project Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
VIDEOS_DIR = ASSETS_DIR / "videos"
IMAGES_DIR = ASSETS_DIR / "images"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = DATA_DIR / "models"

# --- YOLO Model Settings ---
# High-precision popularized YOLOv8 model (prefers YOLOv8s for superior accuracy, with yolov8n fallback)
_yolo_s = MODELS_DIR / "yolov8s.pt"
_yolo_n = MODELS_DIR / "yolov8n.pt"
YOLO_MODEL_PATH = str(_yolo_s if _yolo_s.exists() else _yolo_n)
HELMET_MODEL_PATH = str(MODELS_DIR / "helmet_best.pt")
ACCIDENT_MODEL_PATH = str(MODELS_DIR / "accident_best.pt")

DETECTION_CONFIDENCE = 0.28
IOU_THRESHOLD = 0.45

VEHICLE_CLASSES = {
    0: "Motorcycle",  # motorcycle/scooter rider
    1: "Motorcycle",  # bicycle / moped / two-wheeler
    2: "Car",         # passenger car / sedan / suv / taxi
    3: "Motorcycle",  # motorcycle / scooter
    5: "Bus",         # city bus / transit / coach
    7: "Truck",       # heavy truck / pickup / trailer / lorry
}

# --- Tracker Settings ---
TRACKER_TYPE = "bytetrack"
COUNTING_LINE_RATIO = 0.55

# --- Night Mode / Low-Light Enhancement ---
NIGHT_MODE_AUTO = True
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_SIZE = (8, 8)
BRIGHTNESS_THRESHOLD = 70

# --- Database Settings ---
USE_SQLITE = True
SQLITE_DB_PATH = str(BASE_DIR / "data" / "traffic_data.db")

MYSQL_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "",
    "database": "traffic_signal_db",
}

# --- Adaptive Signal Controller ---
SIGNAL_SETTINGS = {
    "min_green_time": 10,
    "max_green_time": 60,
    "default_green_time": 30,
    "yellow_time": 3,
    "all_red_time": 2,
    "emergency_priority": True,
    "density_factor": 1.5,
}

NUM_LANES = 4

# --- Dashboard Settings ---
DASHBOARD_HOST = "localhost"
DASHBOARD_PORT = 8501
DASHBOARD_TITLE = "Traffic Signal Optimization - AI Dashboard"

TIME_FILTERS = {
    "Last 1 Hour": 1,
    "Last 2 Hours": 2,
    "Last 6 Hours": 6,
    "Last 12 Hours": 12,
    "Last 24 Hours": 24,
}

# --- Export Settings ---
EXPORT_DIR = BASE_DIR / "data"
CSV_EXPORT_FILENAME = "traffic_report.csv"
EXCEL_EXPORT_FILENAME = "traffic_report.xlsx"
