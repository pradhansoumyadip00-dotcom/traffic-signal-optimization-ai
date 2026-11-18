"""
Intelligent Helmet Violation Detector Module
============================================
Detects two-wheeler riders without safety helmets using:
1. Neural Model (if custom helmet_best.pt exists)
2. Computer Vision head-region chromatic and luminance inspection for motorcycle riders
"""

import os
import cv2
import numpy as np
from pathlib import Path

try:
    from config.settings import HELMET_MODEL_PATH
except ImportError:
    HELMET_MODEL_PATH = None


class HelmetDetector:
    """Detects motorcycle helmet compliance and safety violations."""

    def __init__(self):
        self.class_names = {0: "With Helmet", 1: "Without Helmet"}
        self.model = None

        if HELMET_MODEL_PATH and os.path.exists(HELMET_MODEL_PATH):
            try:
                from ultralytics import YOLO
                self.model = YOLO(HELMET_MODEL_PATH)
            except Exception:
                self.model = None

    def detect(self, frame: np.ndarray, tracked_objects: list = None) -> list:
        """
        Detect helmet infractions on two-wheelers.
        
        Args:
            frame: Video frame (np.ndarray)
            tracked_objects: Optional list of tracked vehicles
            
        Returns:
            List of violations: [{"bbox": [x1, y1, x2, y2], "confidence": float, "track_id": int, "violation": "No Helmet"}]
        """
        if frame is None:
            return []

        violations = []

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
                        # Assuming class 1 = Without Helmet
                        if cls_id == 1 and conf > 0.40:
                            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                            violations.append({
                                "bbox": [x1, y1, x2, y2],
                                "confidence": round(conf, 2),
                                "violation": "No Helmet",
                                "track_id": 0
                            })
            except Exception:
                pass

        # 2. Vision heuristic inspection on tracked motorcycles
        if tracked_objects:
            h, w = frame.shape[:2]
            for obj in tracked_objects:
                if obj.get("class") == "Motorcycle":
                    bx1, by1, bx2, by2 = obj["bbox"]
                    bw = bx2 - bx1
                    bh = by2 - by1

                    if bw < 20 or bh < 30:
                        continue

                    # Head region is approximately the top 22% of rider profile
                    hy1 = max(0, by1)
                    hy2 = min(h, by1 + int(bh * 0.22))
                    hx1 = max(0, bx1 + int(bw * 0.20))
                    hx2 = min(w, bx2 - int(bw * 0.20))

                    if hy2 > hy1 and hx2 > hx1:
                        head_roi = frame[hy1:hy2, hx1:hx2]
                        if head_roi.size > 0:
                            # Analyze color space in HSV for hair / skin vs helmet reflection
                            hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
                            # Skin / hair tone ranges
                            lower_skin = np.array([0, 20, 40], dtype=np.uint8)
                            upper_skin = np.array([25, 255, 255], dtype=np.uint8)
                            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
                            skin_ratio = np.count_nonzero(skin_mask) / float(head_roi.shape[0] * head_roi.shape[1])

                            # High bare skin / uncovered hair ratio in head region indicates no helmet
                            if skin_ratio > 0.35:
                                conf = min(0.92, round(0.60 + skin_ratio * 0.35, 2))
                                violations.append({
                                    "bbox": [hx1, hy1, hx2, hy2],
                                    "confidence": conf,
                                    "violation": "No Helmet",
                                    "track_id": obj.get("track_id", 0)
                                })

        return violations
