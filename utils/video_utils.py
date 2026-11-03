"""
Video Utilities
================
Helper functions for video file handling,
frame extraction, and drawing overlays.
"""

import cv2
import numpy as np


def open_video(video_path: str) -> cv2.VideoCapture:
    """Open a video file and return VideoCapture object."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    return cap


def get_video_info(cap: cv2.VideoCapture) -> dict:
    """Get basic video properties."""
    return {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }


def draw_bbox(frame, bbox, label, color=(0, 255, 0), thickness=2):
    """Draw a bounding box with label on the frame."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    # Label background
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(
        frame, (x1, y1 - label_size[1] - 10),
        (x1 + label_size[0], y1), color, -1
    )
    cv2.putText(
        frame, label, (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
    )
    return frame


def draw_counting_line(frame, y_position, color=(0, 235, 255), label="Continuous Line"):
    """Draw the prominent continuous vehicle counting line on the frame."""
    width = frame.shape[1]
    # Dark border backing line for high contrast against light or dark pavement
    cv2.line(frame, (0, y_position), (width, y_position), (15, 15, 15), 5)
    # Vibrant glowing continuous radar line
    cv2.line(frame, (0, y_position), (width, y_position), color, 2)

    # Stylish pill badge container for clean readability
    badge_label = f"--- {label} ---"
    (tw, th), _ = cv2.getTextSize(badge_label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
    bx1 = 14
    by1 = max(4, y_position - th - 12)
    bx2 = bx1 + tw + 16
    by2 = max(th + 16, y_position - 2)

    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 1)
    cv2.putText(
        frame, badge_label, (bx1 + 8, by2 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA
    )
    return frame


def draw_stats_overlay(frame, stats: dict):
    """Draw a semi-transparent stats overlay on the frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    # Background box adjusted dynamically to fit all stats entries
    box_bottom = max(160, 20 + len(stats) * 26)
    cv2.rectangle(overlay, (w - 300, 10), (w - 10, box_bottom), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

    y_offset = 35
    for key, value in stats.items():
        text = f"{key}: {value}"
        cv2.putText(
            frame, text, (w - 290, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
        )
        y_offset += 25

    return frame
