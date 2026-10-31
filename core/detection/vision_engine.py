"""
Master Vision Engine (Central Computer Vision Coordinator)
==========================================================
Coordinates vehicle detection, background subtraction, tracking,
night mode enhancement, and database logging in a single, lightweight pipeline.
Uses OpenCV + Background Subtraction + Gaussian Blurring + NumPy.
Zero heavy neural network / PyTorch / Ultralytics overhead.
"""

import os
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

from core.detection.vehicle_detector import VehicleDetector
from core.detection.helmet_detector import HelmetDetector
from core.detection.accident_detector import AccidentDetector
from core.tracking.tracker import VehicleTracker
from database.db_manager import DatabaseManager
from utils.night_enhance import NightEnhancer
from utils.video_utils import draw_bbox, draw_counting_line, draw_stats_overlay
from config.settings import ASSETS_DIR


class VisionEngine:
    """Master computer vision processing engine using OpenCV & NumPy."""

    # UI colors (BGR)
    COLOR_MAP = {
        "Car": (245, 130, 48),          # Orange-Blue
        "Motorcycle": (50, 205, 50),     # Lime Green
        "Bus": (0, 215, 255),           # Amber Gold
        "Truck": (220, 20, 60),         # Crimson
        "Violation": (0, 0, 220),       # Red
        "Accident": (0, 0, 255)         # Bright Red
    }

    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager if db_manager else DatabaseManager()
        self.tracker = VehicleTracker()
        self.detector = VehicleDetector()
        self.helmet_detector = HelmetDetector()
        self.accident_detector = AccidentDetector()
        self.night_enhancer = NightEnhancer()

        # Violation snapshot directory
        self.snapshot_dir = Path(ASSETS_DIR) / "images" / "violations"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.last_accident_alert = None
        self.last_helmet_alert = None

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0, record_to_db: bool = True) -> tuple:
        """
        Processes a single frame from video file, camera stream, or photo upload.

        Returns:
            (annotated_frame, stats_dict)
        """
        if frame is None:
            return frame, {}

        # 1. Low-light / Night mode enhancement
        is_night = self.night_enhancer.is_dark(frame)
        enhanced_frame = self.night_enhancer.enhance(frame) if is_night else frame
        annotated = enhanced_frame.copy()

        # 2. Tracking & Counting via Background Subtraction + Gaussian Blurring
        track_result = self.tracker.track_frame(enhanced_frame, frame_idx=frame_idx)
        tracked_objects = track_result["tracked_objects"]
        new_counts = track_result["new_counts"]

        # 3. Log newly counted transit vehicles to database
        for vehicle in new_counts:
            tid = vehicle["track_id"]
            if record_to_db:
                self.db.log_vehicle(
                    vehicle_type=vehicle["class"],
                    track_id=tid,
                    confidence=vehicle["confidence"],
                    plate_number="",
                    lane_id=vehicle["lane_id"],
                    speed_estimate=vehicle.get("speed", 38.0)
                )

        # 4. Fallback for static single photos (if no moving tracks found)
        if frame_idx <= 1 and len(tracked_objects) == 0:
            static_detections = self.detector.detect(enhanced_frame)
            for idx, det in enumerate(static_detections):
                x1, y1, x2, y2 = det["bbox"]
                cls_name = det["class"]
                color = self.COLOR_MAP.get(cls_name, (0, 255, 0))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, f"{cls_name}", (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if record_to_db and idx < 10:
                    self.db.log_vehicle(
                        vehicle_type=cls_name,
                        track_id=idx + 1,
                        confidence=det["confidence"],
                        plate_number="",
                        lane_id=(idx % 4) + 1,
                        speed_estimate=40.0
                    )
            if static_detections:
                track_result["total_count"] = len(static_detections)
                for det in static_detections:
                    c_name = det["class"]
                    track_result["class_counts"][c_name] = track_result["class_counts"].get(c_name, 0) + 1
                    l_id = (det["class_id"] % 4)
                    track_result["lane_counts"][l_id] = track_result["lane_counts"].get(l_id, 0) + 1

        # 5. Draw Annotations on Output Frame
        line_y = track_result["counting_line_y"]
        draw_counting_line(annotated, line_y, color=(0, 230, 255))

        # Vehicle Bounding Boxes & Badges
        for obj in tracked_objects:
            color = self.COLOR_MAP.get(obj["class"], (0, 255, 0))
            x1, y1, x2, y2 = obj["bbox"]

            # Vehicle Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Badge placement
            badge_text = f"{obj['class']}"
            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            if y1 >= th + 10:
                by1 = y1 - th - 8
                by2 = y1
                ty = y1 - 4
            else:
                by1 = y1
                by2 = y1 + th + 8
                ty = y1 + th + 4

            cv2.rectangle(annotated, (x1, by1), (x1 + tw + 6, by2), color, -1)
            cv2.putText(annotated, badge_text, (x1 + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw HUD Stats Overlay (top right)
        stats = {
            "Total Count": track_result["total_count"],
            **track_result["class_counts"],
            "Night Mode": "ON" if is_night else "OFF",
            "Frame": frame_idx
        }
        draw_stats_overlay(annotated, stats)

        # Assemble summary return dict
        stats_return = {
            "total_count": track_result["total_count"],
            "class_counts": track_result["class_counts"],
            "lane_counts": track_result["lane_counts"],
            "is_night": is_night,
            "active_vehicles": len(tracked_objects) if tracked_objects else track_result["total_count"],
            "last_helmet_alert": None,
            "last_accident_alert": None
        }

        return annotated, stats_return

    def reset(self):
        """Reset state counters and caches."""
        self.tracker.reset()
