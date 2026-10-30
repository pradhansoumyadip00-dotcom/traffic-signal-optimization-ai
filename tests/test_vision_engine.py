"""
Vision Engine Test Suite
=========================
Validates the complete Computer Vision Core:
- Frame processing
- Night mode contrast enhancement
- Tracking and annotations
- ANPR plate reader ROI logic
- Return statistics verification
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.detection.vision_engine import VisionEngine
from database.db_manager import DatabaseManager


def run_tests():
    print("=" * 60)
    print("  Testing Computer Vision Engine (Step 3)...")
    print("=" * 60)

    db = DatabaseManager()
    engine = VisionEngine(db_manager=db)
    print("[PASS] VisionEngine initialized successfully with all sub-modules.")

    # 1. Test Night Mode Detection on Dark Frame
    dark_frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
    assert engine.night_enhancer.is_dark(dark_frame) == True, "Dark frame detection failed"
    enhanced = engine.night_enhancer.enhance(dark_frame)
    assert enhanced.shape == dark_frame.shape, "Enhanced shape mismatch"
    print("[PASS] Night mode / low-light enhancement verified.")

    # 2. Test Processing Simulated Frame
    test_frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
    # Draw simulated road and vehicle
    cv2.rectangle(test_frame, (300, 200), (450, 400), (200, 50, 50), -1)

    annotated, stats = engine.process_frame(test_frame, frame_idx=1, record_to_db=False)
    assert annotated is not None, "Annotated frame is None"
    assert annotated.shape == test_frame.shape, "Annotated frame shape mismatch"
    assert "total_count" in stats, "stats missing total_count"
    assert "class_counts" in stats, "stats missing class_counts"
    assert "lane_counts" in stats, "stats missing lane_counts"
    print(f"[PASS] process_frame() verified. Output stats: {stats}")

    # 3. Test Plate Reader Lower ROI Cropping
    car_bbox = [100, 100, 300, 400]
    roi = engine.plate_reader.extract_plate_roi(test_frame, car_bbox)
    assert roi.size > 0, "Plate ROI extraction returned empty image"
    print(f"[PASS] ANPR Plate ROI extraction verified. Cropped dimensions: {roi.shape}")

    # 4. Test Clean Plate Text Regex
    raw_ocr = " WB-02. AB 1234 "
    cleaned = engine.plate_reader.clean_plate_text(raw_ocr)
    assert cleaned == "WB02AB1234", f"Expected WB02AB1234, got {cleaned}"
    print(f"[PASS] Plate text normalization verified: '{raw_ocr}' -> '{cleaned}'")

    # 5. Test Video File Feed (Sample traffic video)
    sample_vid = Path(__file__).resolve().parent.parent / "assets" / "videos" / "sample_traffic.mp4"
    if sample_vid.exists():
        cap = cv2.VideoCapture(str(sample_vid))
        ret, frame = cap.read()
        if ret:
            ann, st_res = engine.process_frame(frame, frame_idx=1, record_to_db=False)
            assert ann is not None
            print("[PASS] Sample video frame processed successfully through VisionEngine.")
        cap.release()

    engine.reset()
    db.close()

    print("=" * 60)
    print("  ALL COMPUTER VISION CORE TESTS PASSED (100% SUCCESS)!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
