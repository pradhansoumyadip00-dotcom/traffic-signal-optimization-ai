"""
Synthetic Traffic Video Generator
==================================
Creates a realistic test video with simulated road lanes,
moving cars, motorcycles, trucks, and day/night transitions.
Perfect for testing the pipeline immediately without external downloads.
"""

import cv2
import numpy as np
import random
from pathlib import Path


def generate_traffic_video(output_path: str, duration_sec: int = 15, fps: int = 30):
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    lanes_x = [300, 500, 700, 900]

    # Simulated vehicles: [x, y, speed, type, color, size]
    vehicles = []
    types = [
        ("Car", (50, 50, 220), (60, 110)),
        ("Car", (200, 100, 50), (60, 110)),
        ("Bus", (40, 180, 40), (80, 200)),
        ("Truck", (30, 120, 200), (90, 240)),
        ("Motorcycle", (220, 50, 180), (35, 60)),
    ]

    for frame_idx in range(total_frames):
        # Background road (gray)
        frame = np.full((height, width, 3), 50, dtype=np.uint8)

        # Draw road margins and lanes
        cv2.rectangle(frame, (200, 0), (1080, height), (70, 70, 70), -1)
        for lx in [400, 640, 880]:
            for y_dash in range(0, height, 40):
                cv2.line(frame, (lx, y_dash), (lx, y_dash + 20), (255, 255, 255), 3)

        # Periodically spawn vehicles
        if frame_idx % 25 == 0 and len(vehicles) < 8:
            lane = random.choice(lanes_x)
            v_type, color, size = random.choice(types)
            speed = random.randint(4, 9)
            vehicles.append({
                "x": lane - size[0] // 2,
                "y": -size[1],
                "speed": speed,
                "type": v_type,
                "color": color,
                "w": size[0],
                "h": size[1]
            })

        # Update and draw vehicles
        active_vehicles = []
        for v in vehicles:
            v["y"] += v["speed"]
            vx, vy, vw, vh = v["x"], v["y"], v["w"], v["h"]

            if vy < height + 50:
                # Vehicle body
                cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), v["color"], -1)
                cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), (20, 20, 20), 2)

                # Windshield
                cv2.rectangle(frame, (vx + 5, vy + 20), (vx + vw - 5, vy + 45), (180, 220, 240), -1)

                # Headlights
                cv2.circle(frame, (vx + 10, vy + vh - 10), 6, (0, 255, 255), -1)
                cv2.circle(frame, (vx + vw - 10, vy + vh - 10), 6, (0, 255, 255), -1)

                active_vehicles.append(v)
        vehicles = active_vehicles

        # Night effect simulation on later frames
        if frame_idx > total_frames * 0.7:
            alpha = 0.4
            frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)

        out.write(frame)

    out.release()
    print(f"Test video generated successfully at: {output_path}")


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent.parent / "assets" / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "sample_traffic.mp4"
    generate_traffic_video(str(out_file), duration_sec=10, fps=25)
