"""
Master Vision Engine (Central Computer Vision Coordinator)
==========================================================
Coordinates vehicle detection, tracking, night mode enhancement,
and database logging in a unified, high-performance pipeline.
Uses High-Precision YOLOv8s Neural Model with Adaptive Morphology Fallback.
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
    """Master computer vision processing engine using YOLOv8s & OpenCV."""

    # UI colors (BGR)
    COLOR_MAP = {
        "Car": (245, 130, 48),          # Orange-Blue
        "Motorcycle": (50, 205, 50),     # Lime Green
        "Bus": (0, 215, 255),           # Amber Gold
        "Truck": (220, 20, 60),         # Crimson
        "Violation": (0, 0, 220),       # Red
        "Accident": (0, 0, 255)         # Bright Red
    }

    def __init__(self, db_manager: DatabaseManager = None, video_source: str = "Video 1", line_ratio: float = None):
        self.db = db_manager if db_manager else DatabaseManager()
        self.video_source = video_source
        self.line_ratio = line_ratio
        self.detector = VehicleDetector()
        self.tracker = VehicleTracker(detector=self.detector, line_ratio=self.line_ratio)
        self.helmet_detector = HelmetDetector()
        self.accident_detector = AccidentDetector()
        self.night_enhancer = NightEnhancer()

        # Violation snapshot directory
        self.snapshot_dir = Path(ASSETS_DIR) / "images" / "violations"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        self.last_accident_alert = None
        self.last_helmet_alert = None

    def set_video_source(self, video_source: str):
        """Set the active video source/session label for database logging."""
        self.video_source = video_source

    def reset(self, line_ratio: float = None):
        """Reset state counters and caches."""
        if not hasattr(self, "detector") or self.detector is None:
            self.detector = VehicleDetector()
        if not hasattr(self, "tracker") or self.tracker is None:
            self.tracker = VehicleTracker(detector=self.detector, line_ratio=line_ratio)
        else:
            if not hasattr(self.tracker, "detector") or self.tracker.detector is None:
                self.tracker.detector = self.detector
            self.tracker.reset(line_ratio=line_ratio)

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0, record_to_db: bool = True, video_source: str = None, line_ratio: float = None) -> tuple:
        """
        Processes a single frame from video file, camera stream, or photo upload.

        Returns:
            (annotated_frame, stats_dict)
        """
        if frame is None:
            return frame, {}

        if not hasattr(self, "detector") or self.detector is None:
            self.detector = VehicleDetector()
        if not hasattr(self, "tracker") or self.tracker is None:
            self.tracker = VehicleTracker(detector=self.detector, line_ratio=line_ratio)
        elif not hasattr(self.tracker, "detector") or self.tracker.detector is None:
            self.tracker.detector = self.detector

        # 1. Low-light / Night mode enhancement
        is_night = self.night_enhancer.is_dark(frame)
        enhanced_frame = self.night_enhancer.enhance(frame) if is_night else frame
        annotated = enhanced_frame.copy()

        # 2. Tracking & Counting via Hybrid Tracker (YOLOv8s + IoU + Velocity)
        track_result = self.tracker.track_frame(enhanced_frame, frame_idx=frame_idx, line_ratio=line_ratio)
        tracked_objects = track_result["tracked_objects"]
        new_counts = track_result["new_counts"]

        v_source = video_source or self.video_source or "Video 1"

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
                    speed_estimate=vehicle.get("speed", 38.0),
                    video_source=v_source
                )

        # 4. Fallback for static single photos or frame_idx <= 1:
        if frame_idx <= 1:
            if len(tracked_objects) == 0:
                static_detections = self.detector.detect(enhanced_frame)
                for idx, det in enumerate(static_detections):
                    x1, y1, x2, y2 = det["bbox"]
                    cls_name = det["class"]
                    color = self.COLOR_MAP.get(cls_name, (0, 255, 0))
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    
                    # Clean vehicle class label only (no percentage)
                    badge_text = f"{cls_name}"
                    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    by1 = max(0, y1 - th - 6)
                    cv2.rectangle(annotated, (x1, by1), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(annotated, badge_text, (x1 + 3, max(12, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

                    if record_to_db:
                        self.db.log_vehicle(
                            vehicle_type=cls_name,
                            track_id=idx + 1,
                            confidence=det["confidence"],
                            plate_number="",
                            lane_id=(idx % 4) + 1,
                            speed_estimate=38.0,
                            video_source=v_source
                        )
                if static_detections:
                    track_result["total_count"] = len(static_detections)
                    photo_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
                    for det in static_detections:
                        c_name = det["class"]
                        photo_counts[c_name] = photo_counts.get(c_name, 0) + 1
                        l_id = (det.get("class_id", 0) % 4)
                        track_result["lane_counts"][l_id] = track_result["lane_counts"].get(l_id, 0) + 1
                    track_result["class_counts"] = photo_counts
                    track_result["active_class_counts"] = photo_counts.copy()
            else:
                if record_to_db:
                    for obj in tracked_objects:
                        tid = obj.get("track_id", 1)
                        if tid not in self.tracker.counted_ids:
                            self.db.log_vehicle(
                                vehicle_type=obj["class"],
                                track_id=tid,
                                confidence=obj["confidence"],
                                plate_number="",
                                lane_id=obj["lane_id"] + 1,
                                speed_estimate=obj.get("speed", 38.0),
                                video_source=v_source
                            )

        # 5. Draw Annotations on Output Frame
        line_y = track_result["counting_line_y"]
        draw_counting_line(annotated, line_y, color=(0, 230, 255))

        # Vehicle Bounding Boxes & Badges
        for obj in tracked_objects:
            color = self.COLOR_MAP.get(obj["class"], (0, 255, 0))
            x1, y1, x2, y2 = obj["bbox"]

            # Visual feedback on line touch: highlight box in vibrant green upon touching/counting
            if obj.get("just_counted", False) or (y1 <= line_y <= y2 and obj.get("counted", False)):
                box_color = (0, 255, 0)
                box_thick = 3
            else:
                box_color = color
                box_thick = 2

            # Vehicle Box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, box_thick)

            # Badge placement without percentage (clean vehicle name only)
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
        active_counts = track_result.get("active_class_counts", track_result["class_counts"])
        active_total = len(tracked_objects) if tracked_objects else track_result["total_count"]
        
        # Determine final report numbers (for photos, all visible vehicles count)
        if frame_idx <= 1:
            report_total = max(track_result["total_count"], active_total)
            report_class_counts = active_counts if sum(active_counts.values()) > 0 else track_result["class_counts"]
        else:
            report_total = track_result["total_count"]
            report_class_counts = track_result["class_counts"]

        stats_hud = {
            "Total Logged": report_total,
            "Live in Frame": active_total,
            **active_counts,
            "Night Mode": "ON" if is_night else "OFF",
            "Frame": frame_idx
        }
        draw_stats_overlay(annotated, stats_hud)

        # Assemble summary return dict
        stats_return = {
            "total_count": report_total,
            "class_counts": report_class_counts,
            "active_class_counts": active_counts,
            "lane_counts": track_result["lane_counts"],
            "is_night": is_night,
            "active_vehicles": active_total,
            "last_helmet_alert": None,
            "last_accident_alert": None
        }

        return annotated, stats_return
