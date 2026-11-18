"""
Intelligent Accident & Collision Detector Module
================================================
Detects vehicular collisions and traffic incidents using:
1. Neural Model (if custom accident_best.pt exists)
2. High-precision geometric collision and overlapping box intersection analysis
"""

import os
import cv2
import numpy as np
from pathlib import Path

try:
    from config.settings import ACCIDENT_MODEL_PATH
except ImportError:
    ACCIDENT_MODEL_PATH = None


class AccidentDetector:
    """Detects traffic accidents, collisions, and hazardous stoppages."""

    def __init__(self):
        self.class_names = {0: "Normal", 1: "Accident"}
        self.model = None

        if ACCIDENT_MODEL_PATH and os.path.exists(ACCIDENT_MODEL_PATH):
            try:
                from ultralytics import YOLO
                self.model = YOLO(ACCIDENT_MODEL_PATH)
            except Exception:
                self.model = None

    def detect(self, frame: np.ndarray, tracked_objects: list = None) -> list:
        """
        Detect accidents in current frame.
        
        Args:
            frame: Video frame (np.ndarray)
            tracked_objects: Optional list of currently active tracked objects
            
        Returns:
            List of alerts: [{"bbox": [x1, y1, x2, y2], "confidence": float, "severity": str, "lane_id": int}]
        """
        if frame is None:
            return []

        alerts = []

        # 1. Neural model detection if weights available
        if self.model is not None:
            try:
                results = self.model.predict(source=frame, conf=0.35, verbose=False)
                for res in results:
                    boxes = getattr(res, "boxes", None)
                    if boxes is None:
                        continue
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        if cls_id == 1 or conf > 0.50:
                            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                            alerts.append({
                                "bbox": [x1, y1, x2, y2],
                                "confidence": round(conf, 2),
                                "severity": "High" if conf > 0.70 else "Medium",
                                "lane_id": 0
                            })
            except Exception:
                pass

        # 2. Geometric collision analysis across tracked vehicles
        if tracked_objects and len(tracked_objects) >= 2:
            n = len(tracked_objects)
            for i in range(n):
                for j in range(i + 1, n):
                    o1 = tracked_objects[i]
                    o2 = tracked_objects[j]
                    b1 = o1["bbox"]
                    b2 = o2["bbox"]

                    # Compute intersection
                    xi1 = max(b1[0], b2[0])
                    yi1 = max(b1[1], b2[1])
                    xi2 = min(b1[2], b2[2])
                    yi2 = min(b1[3], b2[3])

                    if xi2 > xi1 and yi2 > yi1:
                        inter_area = (xi2 - xi1) * (yi2 - yi1)
                        area1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
                        area2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
                        min_area = min(area1, area2)
                        overlap_ratio = inter_area / float(min_area)

                        # High overlap between different vehicle types or large overlap ratio
                        if overlap_ratio > 0.42:
                            comb_box = [
                                min(b1[0], b2[0]),
                                min(b1[1], b2[1]),
                                max(b1[2], b2[2]),
                                max(b1[3], b2[3])
                            ]
                            conf = min(0.95, round(0.55 + (overlap_ratio * 0.4), 2))
                            severity = "Critical" if overlap_ratio > 0.65 else "High"
                            lane = o1.get("lane_id", o2.get("lane_id", 0))

                            alerts.append({
                                "bbox": comb_box,
                                "confidence": conf,
                                "severity": severity,
                                "lane_id": lane,
                                "involved_tracks": [o1.get("track_id"), o2.get("track_id")]
                            })

        return alerts
