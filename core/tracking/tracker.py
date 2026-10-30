"""
Vehicle Tracker & Speed Estimation Module
==========================================
Uses YOLOv8 + ByteTrack to track vehicles across frames.
Maintains persistent IDs, detects line crossings without duplicates,
assigns lanes based on X position, and estimates speed.
"""

import torch
from ultralytics import YOLO
import numpy as np
from collections import Counter
from config.settings import (
    YOLO_MODEL_PATH, DETECTION_CONFIDENCE,
    IOU_THRESHOLD, VEHICLE_CLASSES, COUNTING_LINE_RATIO, NUM_LANES
)


class VehicleTracker:
    """
    Advanced multi-object tracker with directional line counting
    and lane assignment.
    """

    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(YOLO_MODEL_PATH)
        self.vehicle_class_ids = list(VEHICLE_CLASSES.keys())

        # State tracking
        self.counted_ids = set()
        self.prev_positions = {}  # {track_id: (center_x, center_y, frame_idx)}
        self.track_class_history = {}  # {track_id: [class_names...]}
        self.total_count = 0
        self.class_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
        self.lane_counts = {i: 0 for i in range(NUM_LANES)}

    def get_lane_id(self, center_x: int, frame_width: int) -> int:
        """Assign lane index (0 to NUM_LANES-1) based on X position."""
        safe_width = max(1, frame_width)
        lane_width = safe_width / max(1, NUM_LANES)
        lane = int(center_x / lane_width)
        return min(max(0, lane), NUM_LANES - 1)

    def estimate_speed(self, track_id: int, current_pos: tuple, frame_idx: int, fps: float = 25.0) -> float:
        """
        Estimate vehicle speed (in km/h) based on pixel displacement.
        """
        if track_id not in self.prev_positions:
            self.prev_positions[track_id] = (current_pos[0], current_pos[1], frame_idx)
            return 35.0  # Default initial reasonable speed

        prev_x, prev_y, prev_frame = self.prev_positions[track_id]
        frame_diff = frame_idx - prev_frame
        if frame_diff <= 0:
            return 35.0

        # Pixel displacement (Euclidean distance)
        dist_px = np.sqrt((current_pos[0] - prev_x)**2 + (current_pos[1] - prev_y)**2)
        # Approximate scale factor: ~15 pixels per meter in standard traffic camera view
        dist_meters = dist_px / 15.0
        time_sec = frame_diff / fps
        speed_mps = dist_meters / max(time_sec, 0.04)
        speed_kmh = speed_mps * 3.6

        # Update position history every 5 frames for stability
        if frame_diff >= 5:
            self.prev_positions[track_id] = (current_pos[0], current_pos[1], frame_idx)

        # Clamp within realistic limits
        return round(float(np.clip(speed_kmh, 15.0, 95.0)), 1)

    def track_frame(self, frame: np.ndarray, frame_idx: int = 0) -> dict:
        """
        Track vehicles in the frame and detect line crossing events.
        """
        frame_height, frame_width = frame.shape[:2]
        counting_line_y = int(frame_height * COUNTING_LINE_RATIO)

        # Run tracking
        results = self.model.track(
            source=frame,
            conf=DETECTION_CONFIDENCE,
            iou=IOU_THRESHOLD,
            classes=self.vehicle_class_ids,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
            device=self.device,
        )

        tracked_objects = []
        new_counts = []

        # 1. Collect raw frame detections and identify bike chassis boxes
        raw_detections = []
        bike_boxes = []

        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue

            for box in boxes:
                if box.id is None:
                    continue
                track_id = int(box.id[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                raw_detections.append((track_id, cls_id, conf, (x1, y1, x2, y2)))
                if cls_id in [1, 3]:  # bicycle or motorcycle
                    bike_boxes.append((x1, y1, x2, y2))

        # 2. Process each track with intelligent rider-to-bike fusion
        for track_id, cls_id, conf, (x1, y1, x2, y2) in raw_detections:
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            lane_id = self.get_lane_id(center_x, frame_width)
            speed = self.estimate_speed(track_id, (center_x, center_y), frame_idx)

            if cls_id in VEHICLE_CLASSES and cls_id != 0:
                raw_class = VEHICLE_CLASSES[cls_id]
            elif cls_id == 0:
                # Fused Rider Detection: Check if person is riding in traffic
                is_rider = False
                for bx1, by1, bx2, by2 in bike_boxes:
                    ix1, iy1 = max(x1, bx1), max(y1, by1)
                    ix2, iy2 = min(x2, bx2), min(y2, by2)
                    if ix2 > ix1 and iy2 > iy1:
                        is_rider = True
                        break
                # If rider on road moving at vehicular speed (>= 12 km/h)
                if not is_rider and speed >= 12.0:
                    is_rider = True

                if is_rider:
                    raw_class = "Motorcycle"
                else:
                    # Slow pedestrian on margin, skip counting as a vehicle
                    continue
            else:
                continue

            if track_id not in self.track_class_history:
                self.track_class_history[track_id] = []
            self.track_class_history[track_id].append(raw_class)

            # Majority voting for stable classification
            confirmed_class = Counter(self.track_class_history[track_id]).most_common(1)[0][0]

            obj = {
                "track_id": track_id,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "class": confirmed_class,
                "confidence": round(conf, 3),
                "center": (center_x, center_y),
                "lane_id": lane_id,
                "speed": speed,
            }
            tracked_objects.append(obj)

            # Advanced multi-condition counting gate (Zero-Miss Trajectory + Straddle)
            if track_id not in self.counted_ids:
                is_stable = len(self.track_class_history[track_id]) >= 2
                prev_pos = self.prev_positions.get(track_id)
                crossed = False

                # 1. Trajectory segment crossing (downward or upward direction)
                if is_stable and prev_pos is not None:
                    prev_y = prev_pos[1]
                    if (prev_y < counting_line_y <= center_y) or (prev_y > counting_line_y >= center_y):
                        crossed = True

                # 2. Bounding box geometric straddle (line is inside vehicle bbox)
                if not crossed and is_stable and (y1 <= counting_line_y <= y2):
                    crossed = True

                # 3. Dynamic proximity buffer (catches corner or fast entries)
                buffer_px = max(26, int(frame_height * 0.04))
                if not crossed and is_stable and abs(center_y - counting_line_y) <= buffer_px:
                    crossed = True

                if crossed:
                    self.counted_ids.add(track_id)
                    self.total_count += 1
                    self.class_counts[confirmed_class] = self.class_counts.get(confirmed_class, 0) + 1
                    self.lane_counts[lane_id] = self.lane_counts.get(lane_id, 0) + 1
                    new_counts.append(obj)

        return {
            "tracked_objects": tracked_objects,
            "new_counts": new_counts,
            "total_count": self.total_count,
            "class_counts": dict(self.class_counts),
            "lane_counts": dict(self.lane_counts),
            "counting_line_y": counting_line_y,
        }

    def reset(self):
        """Reset counters and history."""
        self.counted_ids.clear()
        self.prev_positions.clear()
        self.track_class_history.clear()
        self.total_count = 0
        self.class_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
        self.lane_counts = {i: 0 for i in range(NUM_LANES)}
