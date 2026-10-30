"""
Master Vision Engine (Central AI Coordinator)
==============================================
Orchestrates vehicle detection, ByteTrack tracking, ANPR plate reading,
helmet verification, accident alerts, night mode enhancement, and
database logging in a single, unified pipeline.
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
from core.anpr.plate_reader import PlateReader
from database.db_manager import DatabaseManager
from utils.night_enhance import NightEnhancer
from utils.video_utils import draw_bbox, draw_counting_line, draw_stats_overlay
from config.settings import ASSETS_DIR


class VisionEngine:
    """Master computer vision processing engine."""

    # Class-specific UI colors (BGR)
    COLOR_MAP = {
        "Car": (245, 130, 48),          # Orange-Blue
        "Motorcycle": (50, 205, 50),     # Lime Green
        "Bus": (0, 215, 255),           # Amber Gold
        "Truck": (220, 20, 60),         # Crimson
        "Plate": (255, 255, 255),       # White
        "Violation": (0, 0, 220),       # Red
        "Accident": (0, 0, 255)         # Bright Red
    }

    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager if db_manager else DatabaseManager()
        self.tracker = VehicleTracker()
        self.helmet_detector = HelmetDetector()
        self.accident_detector = AccidentDetector()
        self.plate_reader = PlateReader()
        self.night_enhancer = NightEnhancer()

        # Violation snapshot directory
        self.snapshot_dir = Path(ASSETS_DIR) / "images" / "violations"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.last_accident_alert = None
        self.last_helmet_alert = None

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0, record_to_db: bool = True) -> tuple:
        """
        Processes a single frame from video file or live webcam.

        Returns:
            (annotated_frame, stats_dict)
        """
        # 1. Low-light / Night mode enhancement
        is_night = self.night_enhancer.is_dark(frame)
        enhanced_frame = self.night_enhancer.enhance(frame) if is_night else frame
        annotated = enhanced_frame.copy()

        # 2. Tracking & Counting
        track_result = self.tracker.track_frame(enhanced_frame, frame_idx=frame_idx)
        tracked_objects = track_result["tracked_objects"]
        new_counts = track_result["new_counts"]

        # 3. Process Tracked Vehicles (Plate OCR & DB Logging)
        for vehicle in tracked_objects:
            tid = vehicle["track_id"]
            if tid not in self.plate_reader.plate_cache:
                try:
                    self.plate_reader.process_vehicle(
                        enhanced_frame, vehicle["bbox"], tid, vehicle["class"]
                    )
                except Exception:
                    pass

        for vehicle in new_counts:
            tid = vehicle["track_id"]
            plate_text = self.plate_reader.plate_cache.get(tid, "")
            if not plate_text:
                try:
                    plate_text = self.plate_reader.process_vehicle(
                        enhanced_frame, vehicle["bbox"], tid, vehicle["class"]
                    )
                except Exception:
                    pass

            if record_to_db:
                self.db.log_vehicle(
                    vehicle_type=vehicle["class"],
                    track_id=tid,
                    confidence=vehicle["confidence"],
                    plate_number=plate_text,
                    lane_id=vehicle["lane_id"],
                    speed_estimate=vehicle.get("speed", 40.0)
                )

        # 4. Periodic Helmet Violation Check (every 10 frames)
        helmet_violations_found = []
        if frame_idx % 10 == 0:
            try:
                helmet_results = self.helmet_detector.detect(enhanced_frame)
                for h in helmet_results:
                    if h["is_violation"]:
                        helmet_violations_found.append(h)
                        # Save violation snapshot
                        snap_name = f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                        snap_path = str(self.snapshot_dir / snap_name)
                        cv2.imwrite(snap_path, enhanced_frame)

                        if record_to_db:
                            self.db.log_helmet_violation(
                                confidence=h["confidence"],
                                snapshot_path=snap_path
                            )
                        self.last_helmet_alert = snap_name
            except Exception:
                pass

        # 5. Periodic Accident Check (every 15 frames)
        accidents_found = []
        if frame_idx % 15 == 0:
            try:
                accident_results = self.accident_detector.detect(enhanced_frame)
                for a in accident_results:
                    if a["is_accident"]:
                        accidents_found.append(a)
                        snap_name = f"accident_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                        snap_path = str(self.snapshot_dir / snap_name)
                        cv2.imwrite(snap_path, enhanced_frame)

                        if record_to_db:
                            self.db.log_accident(
                                confidence=a["confidence"],
                                severity="High" if a["confidence"] > 0.75 else "Medium",
                                snapshot_path=snap_path
                            )
                        self.last_accident_alert = snap_name
            except Exception:
                pass

        # 6. Draw Annotations on Output Frame
        # Counting Line
        line_y = track_result["counting_line_y"]
        draw_counting_line(annotated, line_y, color=(0, 230, 255))

        # Vehicle Bounding Boxes & Badges
        for obj in tracked_objects:
            color = self.COLOR_MAP.get(obj["class"], (0, 255, 0))
            x1, y1, x2, y2 = obj["bbox"]

            # Vehicle Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Smart non-overlapping badge placement
            badge_text = f"#{obj['track_id']} {obj['class']} {obj['speed']}km/h"
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

            # Draw plate tag if cached or read
            cached_plate = self.plate_reader.plate_cache.get(obj["track_id"])
            if not cached_plate:
                try:
                    cached_plate = self.plate_reader.process_vehicle(enhanced_frame, obj["bbox"], obj["track_id"], obj.get("class", "Car"))
                except Exception:
                    pass
            if cached_plate:
                plate_badge = f"PLATE: {cached_plate}"
                (pw, ph), _ = cv2.getTextSize(plate_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (x1, y2), (x1 + pw + 10, y2 + ph + 8), (15, 15, 15), -1)
                cv2.rectangle(annotated, (x1, y2), (x1 + pw + 10, y2 + ph + 8), (0, 220, 255), 1)
                cv2.putText(annotated, plate_badge, (x1 + 4, y2 + ph + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

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
            "active_vehicles": len(tracked_objects),
            "last_helmet_alert": self.last_helmet_alert,
            "last_accident_alert": self.last_accident_alert
        }

        return annotated, stats_return

    def reset(self):
        """Reset state counters and caches."""
        self.tracker.reset()
        self.plate_reader.plate_cache.clear()
