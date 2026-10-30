"""
Vehicle Detection Module
========================
Detects and classifies vehicles (Car, Motorcycle, Bus, Truck)
using YOLOv8 with automatic GPU/CPU device selection.
"""

import torch
from ultralytics import YOLO
import cv2
import numpy as np
from config.settings import (
    YOLO_MODEL_PATH, DETECTION_CONFIDENCE,
    IOU_THRESHOLD, VEHICLE_CLASSES
)


class VehicleDetector:
    """Real-time vehicle detection using YOLOv8."""

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(YOLO_MODEL_PATH)
        self.vehicle_class_ids = list(VEHICLE_CLASSES.keys())

    def detect(self, frame: np.ndarray) -> list:
        """
        Detect vehicles in a single video frame.

        Args:
            frame: BGR image (numpy array from OpenCV)

        Returns:
            List of dicts: [{"bbox": [x1,y1,x2,y2], "class": "Car", "confidence": 0.92}, ...]
        """
        results = self.model.predict(
            source=frame,
            conf=DETECTION_CONFIDENCE,
            iou=IOU_THRESHOLD,
            classes=self.vehicle_class_ids,
            verbose=False,
            device=self.device,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class": VEHICLE_CLASSES.get(cls_id, "Unknown"),
                    "class_id": cls_id,
                    "confidence": round(conf, 3),
                })
        return detections
