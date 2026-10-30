"""
Helmet Detection Module
========================
Detects whether motorcycle riders are wearing helmets.
Uses a fine-tuned YOLOv8 model with auto GPU/CPU selection
and graceful fallback to base model if weights are not yet downloaded.
"""

import os
import torch
from ultralytics import YOLO
import numpy as np
from config.settings import HELMET_MODEL_PATH, YOLO_MODEL_PATH, DETECTION_CONFIDENCE


class HelmetDetector:
    """Detects helmet violations on motorcycle riders."""

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        model_path = HELMET_MODEL_PATH if os.path.exists(HELMET_MODEL_PATH) else YOLO_MODEL_PATH
        self.model = YOLO(model_path)
        self.class_names = {0: "With Helmet", 1: "Without Helmet"}

    def detect(self, frame: np.ndarray) -> list:
        results = self.model.predict(
            source=frame,
            conf=DETECTION_CONFIDENCE,
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
                    "label": self.class_names.get(cls_id, "Without Helmet"),
                    "is_violation": cls_id == 1 or cls_id == 3,  # motorcycle / violation
                    "confidence": round(conf, 3),
                })
        return detections
