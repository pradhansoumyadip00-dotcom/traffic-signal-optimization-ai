"""
Traffic Signal Optimization on Agent
======================================
Main pipeline entry point supporting:
1. Video file input (--video path/to/video.mp4)
2. Live Webcam feed (--webcam)

Usage:
    python main.py --video assets/videos/sample_traffic.mp4
    python main.py --webcam
"""

import argparse
import cv2
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.detection.vision_engine import VisionEngine
from core.agent.signal_controller import AdaptiveSignalController
from database.db_manager import DatabaseManager


def run_pipeline(source=0, is_webcam: bool = False, show_display: bool = True, max_frames: int = 0):
    print("=" * 60)
    print("  Traffic Signal Optimization on Agent")
    print(f"  Mode: {'Live Webcam' if is_webcam else f'Video File ({source})'}")
    print("=" * 60)

    # Initialize components
    db = DatabaseManager()
    vision_engine = VisionEngine(db_manager=db)
    signal_controller = AdaptiveSignalController(db_manager=db)

    cap = cv2.VideoCapture(0 if is_webcam else source)
    if not cap.isOpened():
        print(f"[Error] Could not open video source: {source}")
        db.close()
        return

    frame_count = 0
    stats = {
        "total_count": 0,
        "class_counts": {},
        "lane_counts": {i: 0 for i in range(4)}
    }
    print("  AI Pipeline Running... Press 'q' in video window to exit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if 0 < max_frames < frame_count:
                break

            # Run master vision engine on frame
            annotated_frame, stats = vision_engine.process_frame(
                frame, frame_idx=frame_count, record_to_db=True
            )

            # Update adaptive signal agent with lane vehicle counts
            for lane_id, count in stats.get("lane_counts", {}).items():
                signal_controller.update_density(lane_id, count)

            # Advance agent simulation clock every ~30 frames (1 second of video)
            if frame_count % 30 == 0:
                signal_controller.tick_second()

            # Display output
            if show_display:
                try:
                    cv2.imshow("Traffic Signal Optimization — AI Vision", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except Exception:
                    pass
    finally:
        # Cleanup
        cap.release()
        if show_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        db.close()

    print("\n" + "=" * 60)
    print(f"  Pipeline Finished. Total frames processed: {frame_count}")
    print(f"  Total vehicles detected and logged: {stats.get('total_count', 0)}")
    print(f"  Vehicle breakdown: {stats.get('class_counts', {})}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic Signal Optimization on Agent")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Path to traffic video file")
    group.add_argument("--webcam", action="store_true", help="Use live laptop webcam feed")
    parser.add_argument("--no-display", action="store_true", help="Run in headless mode without window")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum frames to process (0 = infinite/whole video)")
    args = parser.parse_args()

    if args.webcam:
        run_pipeline(source=0, is_webcam=True, show_display=not args.no_display, max_frames=args.max_frames)
    else:
        run_pipeline(source=args.video, is_webcam=False, show_display=not args.no_display, max_frames=args.max_frames)
