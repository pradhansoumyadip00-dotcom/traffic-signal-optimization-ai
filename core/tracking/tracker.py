"""
Advanced Duplicate-Proof Vehicle Tracker (YOLOv8s + Spatial Line Gate + Continuous Touch Detection)
==================================================================================================
Coordinates multi-object tracking utilizing IoU & Centroid Association, Class Smoothing,
and a strict Single-Touch Continuous Line Crossing Gate with Spatial Memory.
Prevents duplicate counts: each vehicle is counted EXACTLY ONCE when it touches the line.
"""

from collections import Counter
import cv2
import numpy as np
from config.settings import COUNTING_LINE_RATIO, NUM_LANES
from core.detection.vehicle_detector import VehicleDetector


def calculate_iou(boxA, boxB):
    """Compute Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union_area = float(areaA + areaB - inter_area)

    return inter_area / union_area if union_area > 0 else 0.0


class VehicleTracker:
    """
    Precision vehicle tracker with duplicate-proof continuous line counting.
    Only counts a vehicle when its bounding box/trajectory touches the continuous radar line.
    """

    def __init__(self, detector=None, line_ratio: float = None):
        self.detector = detector if detector is not None else VehicleDetector()
        self.line_ratio = float(line_ratio) if line_ratio is not None else COUNTING_LINE_RATIO
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # Tracks registry: dict(track_id -> track_data)
        self.tracks = {}
        self.next_track_id = 1
        self.counted_ids = set()

        # Spatial memory of counted vehicles to prevent duplicates on ID-switching or flicker
        # Format: [(frame_idx, center_x, center_y, x1, x2)]
        self.recent_counted_spatially = []

        # Cumulative counters (counted ONLY upon touching continuous line)
        self.total_count = 0
        self.class_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
        self.lane_counts = {i: 0 for i in range(NUM_LANES)}

    def reset(self, line_ratio: float = None):
        """Reset all tracking states, spatial memory, and cumulative counts."""
        if not hasattr(self, "detector") or self.detector is None:
            self.detector = VehicleDetector()
        if line_ratio is not None:
            self.line_ratio = float(line_ratio)
        self.tracks.clear()
        self.next_track_id = 1
        self.counted_ids.clear()
        self.recent_counted_spatially.clear()
        self.total_count = 0
        self.class_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
        self.lane_counts = {i: 0 for i in range(NUM_LANES)}

    def get_lane_id(self, center_x: int, frame_width: int) -> int:
        """Assign lane index (0 to NUM_LANES-1) based on horizontal position."""
        safe_width = max(1, frame_width)
        lane_width = safe_width / max(1, NUM_LANES)
        lane = int(center_x / lane_width)
        return min(max(0, lane), NUM_LANES - 1)

    def estimate_speed(self, prev_pos: tuple, curr_pos: tuple, frame_diff: int, fps: float = 25.0) -> float:
        """Estimate realistic road vehicle speed in km/h."""
        if frame_diff <= 0:
            return 38.0
        dist_px = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
        dist_meters = dist_px / 14.0
        time_sec = frame_diff / fps
        speed_mps = dist_meters / max(time_sec, 0.04)
        speed_kmh = speed_mps * 3.6
        return round(float(np.clip(speed_kmh, 20.0, 80.0)), 1)

    def track_frame(self, frame: np.ndarray, frame_idx: int = 0, line_ratio: float = None) -> dict:
        """
        Process frame with High-Precision YOLO Detection and Duplicate-Proof Line Tracking.
        """
        if not hasattr(self, "detector") or self.detector is None:
            self.detector = VehicleDetector()

        frame_height, frame_width = frame.shape[:2]
        effective_ratio = float(line_ratio) if line_ratio is not None else getattr(self, "line_ratio", COUNTING_LINE_RATIO)
        counting_line_y = int(frame_height * effective_ratio)
        line_band = max(8, int(frame_height * 0.025))

        # 1. Clean Detections via Neural Detector
        dets = self.detector.detect(frame)
        current_detections = []

        if dets:
            for d in dets:
                bx1, by1, bx2, by2 = d["bbox"]
                cx = (bx1 + bx2) // 2
                cy = (by1 + by2) // 2
                lane_id = self.get_lane_id(cx, frame_width)
                current_detections.append({
                    "bbox": [bx1, by1, bx2, by2],
                    "centroid": (cx, cy),
                    "class": d["class"],
                    "lane_id": lane_id,
                    "confidence": d["confidence"],
                    "area": (bx2 - bx1) * (by2 - by1)
                })

        # 2. Tracking Association: IoU & Distance
        track_ids = list(self.tracks.keys())
        assigned_det_idx = set()
        updated_track_ids = set()
        new_counts = []

        if len(track_ids) > 0 and len(current_detections) > 0:
            # Match each existing track to best detection
            for tid in track_ids:
                track = self.tracks[tid]
                best_det_idx = None
                best_cost = 9999.0

                tcx, tcy = track["centroid"]
                tbw = track["bbox"][2] - track["bbox"][0]
                tbh = track["bbox"][3] - track["bbox"][1]
                max_match_dist = max(70.0, float(max(tbw, tbh)) * 1.6)

                for c, det in enumerate(current_detections):
                    if c in assigned_det_idx:
                        continue
                    dcx, dcy = det["centroid"]
                    dist = float(np.sqrt((tcx - dcx)**2 + (tcy - dcy)**2))
                    iou = calculate_iou(track["bbox"], det["bbox"])

                    # Cost combining IoU overlap and centroid proximity
                    cost = (1.0 - iou) * 45.0 + min(dist, 90.0)

                    if (dist < max_match_dist or iou > 0.15) and cost < best_cost:
                        best_cost = cost
                        best_det_idx = c

                if best_det_idx is not None and best_cost < 95.0:
                    det = current_detections[best_det_idx]
                    assigned_det_idx.add(best_det_idx)
                    updated_track_ids.add(tid)

                    vx = det["centroid"][0] - track["centroid"][0]
                    vy = det["centroid"][1] - track["centroid"][1]
                    frame_diff = frame_idx - track["last_frame"]
                    speed = self.estimate_speed(track["centroid"], det["centroid"], max(1, frame_diff))

                    hits = track["hits"] + 1
                    track["history"].append(det["class"])
                    confirmed_class = Counter(track["history"]).most_common(1)[0][0]

                    det_y1, det_y2 = det["bbox"][1], det["bbox"][3]
                    old_cy = track["centroid"][1]
                    new_cx, new_cy = det["centroid"]

                    # 3. HIGH-PRECISION CONTINUOUS LINE TOUCH & CROSSING DETECTION
                    # Vehicle touches the continuous radar band or crosses its y-boundary
                    touches_line = (
                        (det_y1 <= (counting_line_y + line_band) and det_y2 >= (counting_line_y - line_band))
                        or (old_cy <= counting_line_y and new_cy >= counting_line_y)
                        or (old_cy >= counting_line_y and new_cy <= counting_line_y)
                        or (abs(new_cy - counting_line_y) <= line_band)
                    )

                    just_counted_flag = False

                    if touches_line and not track["counted"] and hits >= 1:
                        # Spatial duplicate protection: verify no duplicate count in this horizontal corridor
                        is_duplicate = False
                        cur_w = max(1, det["bbox"][2] - det["bbox"][0])
                        for rf, rcx, rcy, rx1, rx2 in self.recent_counted_spatially:
                            if frame_idx - rf < 40:
                                h_overlap = max(0, min(det["bbox"][2], rx2) - max(det["bbox"][0], rx1))
                                if (h_overlap / cur_w > 0.40) or (abs(new_cx - rcx) < max(28, int(cur_w * 0.4))):
                                    is_duplicate = True
                                    break

                        if not is_duplicate:
                            track["counted"] = True
                            just_counted_flag = True
                            self.counted_ids.add(tid)
                            self.total_count += 1
                            self.class_counts[confirmed_class] = self.class_counts.get(confirmed_class, 0) + 1
                            self.lane_counts[det["lane_id"]] = self.lane_counts.get(det["lane_id"], 0) + 1
                            self.recent_counted_spatially.append((frame_idx, new_cx, new_cy, det["bbox"][0], det["bbox"][2]))

                            new_counts.append({
                                "track_id": tid,
                                "class": confirmed_class,
                                "confidence": det["confidence"],
                                "lane_id": det["lane_id"] + 1,
                                "speed": speed
                            })
                        else:
                            # Already counted spatially, mark track as counted to avoid re-evaluating
                            track["counted"] = True

                    track.update({
                        "bbox": det["bbox"],
                        "prev_centroid": track["centroid"],
                        "centroid": det["centroid"],
                        "velocity": (vx, vy),
                        "class": confirmed_class,
                        "confidence": det["confidence"],
                        "lane_id": det["lane_id"],
                        "speed": speed,
                        "hits": hits,
                        "last_frame": frame_idx,
                        "time_since_update": 0,
                        "just_counted": just_counted_flag
                    })

        # 4. Handle unmatched tracks (coasting & pruning)
        stale_tids = []
        for tid, track in self.tracks.items():
            if tid not in updated_track_ids:
                track["time_since_update"] += 1
                track["just_counted"] = False
                track["centroid"] = (
                    track["centroid"][0] + track["velocity"][0],
                    track["centroid"][1] + track["velocity"][1]
                )
                if track["time_since_update"] > 20:
                    stale_tids.append(tid)

        for tid in stale_tids:
            del self.tracks[tid]

        # 5. Clean up old spatial memory entries (> 120 frames)
        if len(self.recent_counted_spatially) > 60:
            self.recent_counted_spatially = [
                entry for entry in self.recent_counted_spatially
                if frame_idx - entry[0] < 120
            ]

        # 6. Register new detections (never count on spawn unless confirmed crossing)
        for c, det in enumerate(current_detections):
            if c not in assigned_det_idx:
                tid = self.next_track_id
                self.next_track_id += 1

                det_y1, det_y2 = det["bbox"][1], det["bbox"][3]
                touches_line = (det_y1 <= (counting_line_y + line_band) and det_y2 >= (counting_line_y - line_band))
                already_counted = False

                if touches_line:
                    new_cx, new_cy = det["centroid"]
                    cur_w = max(1, det["bbox"][2] - det["bbox"][0])
                    for rf, rcx, rcy, rx1, rx2 in self.recent_counted_spatially:
                        if frame_idx - rf < 40:
                            h_overlap = max(0, min(det["bbox"][2], rx2) - max(det["bbox"][0], rx1))
                            if (h_overlap / cur_w > 0.40) or (abs(new_cx - rcx) < max(28, int(cur_w * 0.4))):
                                already_counted = True
                                break

                self.tracks[tid] = {
                    "bbox": det["bbox"],
                    "centroid": det["centroid"],
                    "prev_centroid": det["centroid"],
                    "velocity": (0, 0),
                    "class": det["class"],
                    "history": [det["class"]],
                    "confidence": det["confidence"],
                    "lane_id": det["lane_id"],
                    "speed": 38.0,
                    "hits": 1,
                    "last_frame": frame_idx,
                    "time_since_update": 0,
                    "counted": already_counted,
                    "just_counted": False
                }

        # 7. Collect currently visible active objects on screen
        active_objects = []
        active_class_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
        for tid, track in self.tracks.items():
            if track["time_since_update"] <= 2 and track["hits"] >= 1:
                c_name = track["class"]
                active_class_counts[c_name] = active_class_counts.get(c_name, 0) + 1
                active_objects.append({
                    "track_id": tid,
                    "bbox": track["bbox"],
                    "class": track["class"],
                    "confidence": track["confidence"],
                    "lane_id": track["lane_id"],
                    "speed": track["speed"],
                    "counted": track["counted"],
                    "just_counted": track.get("just_counted", False)
                })

        return {
            "tracked_objects": active_objects,
            "new_counts": new_counts,
            "counting_line_y": counting_line_y,
            "line_ratio": effective_ratio,
            "total_count": self.total_count,
            "class_counts": self.class_counts.copy(),
            "active_class_counts": active_class_counts,
            "lane_counts": self.lane_counts.copy()
        }
