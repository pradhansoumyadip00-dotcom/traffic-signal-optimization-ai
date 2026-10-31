"""
Lightweight Accident Detector Module
====================================
Pure Python/OpenCV lightweight module with zero torch/ultralytics dependencies.
"""

import numpy as np


class AccidentDetector:
    """Lightweight stub for accident detection without neural network overhead."""

    def __init__(self):
        self.class_names = {0: "Normal", 1: "Accident"}

    def detect(self, frame: np.ndarray) -> list:
        """Returns detected accident alerts (empty list for lightweight deployment)."""
        return []
