"""
Night Mode / Low-Light Enhancement
====================================
Enhances dark or night-time video frames so that the
AI models can detect vehicles even in poor lighting.
"""

import cv2
import numpy as np
from config.settings import (
    NIGHT_MODE_AUTO, CLAHE_CLIP_LIMIT,
    CLAHE_TILE_SIZE, BRIGHTNESS_THRESHOLD
)


class NightEnhancer:
    """Enhances low-light video frames for better detection."""

    def __init__(self):
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_TILE_SIZE,
        )

    def is_dark(self, frame: np.ndarray) -> bool:
        """
        Check if a frame is too dark (night-time).

        Args:
            frame: BGR image.

        Returns:
            True if the frame brightness is below threshold.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        return mean_brightness < BRIGHTNESS_THRESHOLD

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE contrast enhancement to a dark frame.

        Args:
            frame: BGR image.

        Returns:
            Enhanced BGR image.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to lightness channel only
        l_enhanced = self.clahe.apply(l_channel)

        # Merge channels back
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

        return result

    def auto_enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Automatically enhance frame if it is dark.
        If NIGHT_MODE_AUTO is True, checks brightness first.
        """
        if NIGHT_MODE_AUTO and self.is_dark(frame):
            return self.enhance(frame)
        return frame
