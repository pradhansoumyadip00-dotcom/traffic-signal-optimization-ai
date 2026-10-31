"""
Lightweight Helmet Detector Module
==================================
Pure Python/OpenCV lightweight module with zero torch/ultralytics dependencies.
"""

import numpy as np


class HelmetDetector:
    """Lightweight stub for helmet detection without neural network overhead."""

    def __init__(self):
        self.class_names = {0: "With Helmet", 1: "Without Helmet"}

    def detect(self, frame: np.ndarray) -> list:
        """Returns detected helmet violations (empty list for lightweight deployment)."""
        return []
