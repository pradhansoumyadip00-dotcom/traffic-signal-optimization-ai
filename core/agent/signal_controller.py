"""
Intelligent Adaptive Traffic Signal Controller (The AI Agent)
==============================================================
Enterprise-grade agentic traffic light controller.
Features:
1. Dynamic density-weighted green time calculation (Webster-style)
2. Starvation prevention (wait-time penalty avoids infinite red)
3. Safe state transitions: GREEN -> YELLOW -> ALL_RED -> GREEN
4. Emergency vehicle override (Ambulance/Fire truck immediate clearance)
5. Night/Off-peak quick-cycle mode
6. Automatic database transition logging
"""

import time
from enum import Enum
from datetime import datetime
from config.settings import SIGNAL_SETTINGS, NUM_LANES
from database.db_manager import DatabaseManager


class SignalState(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ALL_RED = "ALL_RED"
    RED = "RED"


class LaneAgent:
    """Represents a single lane / approach direction at an intersection."""

    LANE_NAMES = {
        0: "North Approach (Lane 1)",
        1: "South Approach (Lane 2)",
        2: "East Approach (Lane 3)",
        3: "West Approach (Lane 4)"
    }

    def __init__(self, lane_id: int):
        self.lane_id = lane_id
        self.name = self.LANE_NAMES.get(lane_id, f"Lane {lane_id + 1}")
        self.state = SignalState.RED
        self.vehicle_count = 0
        self.density = 0.0              # Share of intersection traffic (0.0 - 1.0)
        self.green_time = SIGNAL_SETTINGS["default_green_time"]
        self.remaining_time = 0
        self.waiting_time_sec = 0        # Seconds waiting in RED (starvation tracker)
        self.has_emergency = False

    def to_dict(self) -> dict:
        return {
            "lane_id": self.lane_id,
            "name": self.name,
            "state": self.state.value,
            "vehicle_count": self.vehicle_count,
            "density": round(self.density, 2),
            "green_time": self.green_time,
            "remaining_time": self.remaining_time,
            "waiting_time_sec": self.waiting_time_sec,
            "has_emergency": self.has_emergency,
        }


class AdaptiveSignalController:
    """
    Intelligent Agent managing multi-way traffic signals.
    """

    def __init__(self, db_manager: DatabaseManager = None):
        self.db = db_manager if db_manager else DatabaseManager()
        self.lanes = [LaneAgent(i) for i in range(NUM_LANES)]
        self.current_green_lane = 0
        self.in_transition = False
        self.transition_state = SignalState.GREEN
        self.transition_timer = 0
        self.emergency_lane = None
        self.total_cycles_completed = 0

        # Initialize: Lane 0 starts with GREEN, others RED
        self.lanes[0].state = SignalState.GREEN
        self.lanes[0].green_time = SIGNAL_SETTINGS["default_green_time"]
        self.lanes[0].remaining_time = self.lanes[0].green_time

        # Initial DB log
        self._log_transition(0, "GREEN", self.lanes[0].green_time, 0)

    def update_density(self, lane_id: int, vehicle_count: int, has_emergency: bool = False):
        """Update live traffic counts and emergency status for a lane."""
        if 0 <= lane_id < NUM_LANES:
            self.lanes[lane_id].vehicle_count = max(0, vehicle_count)
            self.lanes[lane_id].has_emergency = has_emergency

            # Recompute density distribution across all lanes
            total_vehicles = sum(l.vehicle_count for l in self.lanes)
            default_density = 1.0 / max(1, NUM_LANES)
            for lane in self.lanes:
                lane.density = (lane.vehicle_count / total_vehicles) if total_vehicles > 0 else default_density

            # If emergency vehicle detected, trigger override
            if has_emergency and self.current_green_lane != lane_id:
                self.trigger_emergency(lane_id)

    def calculate_adaptive_green_time(self, lane_id: int) -> int:
        """
        Calculates optimal green time factoring in:
        1. Vehicle density (pressure)
        2. Wait time (starvation penalty)
        3. Night/Off-peak mode
        """
        lane = self.lanes[lane_id]
        settings = SIGNAL_SETTINGS
        base_time = settings["default_green_time"]
        min_time = settings["min_green_time"]
        max_time = settings["max_green_time"]

        # Total vehicles across intersection
        total_v = sum(l.vehicle_count for l in self.lanes)

        # Off-peak / late-night traffic: if total traffic is very light (< 5 cars)
        if total_v < 5:
            return min_time

        # Starvation penalty: every 30 seconds of waiting adds +2 seconds of green
        starvation_bonus = min(15, int(lane.waiting_time_sec / 30) * 2)

        # Density factor: more vehicles -> more green time
        density_weight = lane.density * settings["density_factor"]
        computed = int(base_time * (1 + density_weight)) + starvation_bonus

        # Clamp between bounds
        return max(min_time, min(max_time, computed))

    def trigger_emergency(self, lane_id: int):
        """
        Emergency vehicle override: Immediately grants priority clearance.
        """
        if not SIGNAL_SETTINGS.get("emergency_priority", True):
            return

        if 0 <= lane_id < NUM_LANES and self.current_green_lane != lane_id:
            self.emergency_lane = lane_id
            # Force current green to yellow immediately
            curr = self.lanes[self.current_green_lane]
            curr.state = SignalState.YELLOW
            curr.remaining_time = SIGNAL_SETTINGS["yellow_time"]
            self.in_transition = True
            self.transition_state = SignalState.YELLOW
            self.transition_timer = SIGNAL_SETTINGS["yellow_time"]
            self._log_transition(self.current_green_lane, "YELLOW", curr.remaining_time, curr.vehicle_count)

    def select_next_lane(self) -> int:
        """
        Fairness-based lane selector:
        Selects next lane considering both vehicle density and wait time.
        """
        if self.emergency_lane is not None:
            chosen = self.emergency_lane
            self.emergency_lane = None
            return chosen

        # Calculate priority score = (density * 100) + (waiting_time_sec / 2)
        candidate_lanes = [i for i in range(NUM_LANES) if i != self.current_green_lane]
        if not candidate_lanes:
            return self.current_green_lane

        best_lane = candidate_lanes[0]
        highest_score = -1.0

        for idx in candidate_lanes:
            lane = self.lanes[idx]
            score = (lane.density * 100.0) + (lane.waiting_time_sec * 0.5)
            if score > highest_score:
                highest_score = score
                best_lane = idx

        return best_lane

    def tick_second(self) -> dict:
        """
        Advances the simulation by 1 second.
        Handles countdowns, safe transitions, and state changes.
        """
        # Update red waiting times for all red lanes
        for lane in self.lanes:
            if lane.state in (SignalState.RED, SignalState.ALL_RED):
                lane.waiting_time_sec += 1

        curr_lane = self.lanes[self.current_green_lane]

        # Handle active transitions (Yellow -> All-Red -> Next Green)
        if self.in_transition:
            self.transition_timer -= 1
            if self.transition_timer <= 0:
                if self.transition_state == SignalState.YELLOW:
                    # Switch to ALL_RED clearance interval
                    self.transition_state = SignalState.ALL_RED
                    self.transition_timer = SIGNAL_SETTINGS["all_red_time"]
                    curr_lane.state = SignalState.ALL_RED
                    self._log_transition(self.current_green_lane, "ALL_RED", self.transition_timer, curr_lane.vehicle_count)
                elif self.transition_state == SignalState.ALL_RED:
                    # Clearance over: Set old lane to RED, switch to Next Green Lane
                    curr_lane.state = SignalState.RED
                    self._log_transition(self.current_green_lane, "RED", 0, curr_lane.vehicle_count)

                    next_lane_id = self.select_next_lane()
                    new_green_time = self.calculate_adaptive_green_time(next_lane_id)

                    next_lane = self.lanes[next_lane_id]
                    next_lane.state = SignalState.GREEN
                    next_lane.green_time = new_green_time
                    next_lane.remaining_time = new_green_time
                    next_lane.waiting_time_sec = 0  # Reset wait time

                    self.current_green_lane = next_lane_id
                    self.in_transition = False
                    self.total_cycles_completed += 1
                    self._log_transition(next_lane_id, "GREEN", new_green_time, next_lane.vehicle_count)
            return self.get_status()

        # Regular GREEN countdown
        curr_lane.remaining_time -= 1
        if curr_lane.remaining_time <= 0:
            # Time to initiate transition: GREEN -> YELLOW
            curr_lane.state = SignalState.YELLOW
            curr_lane.remaining_time = SIGNAL_SETTINGS["yellow_time"]
            self.in_transition = True
            self.transition_state = SignalState.YELLOW
            self.transition_timer = SIGNAL_SETTINGS["yellow_time"]
            self._log_transition(self.current_green_lane, "YELLOW", curr_lane.remaining_time, curr_lane.vehicle_count)

        return self.get_status()

    def _log_transition(self, lane_id: int, state: str, green_time: int, vehicle_count: int):
        """Log state transition to database."""
        try:
            self.db.log_signal_change(
                lane_id=lane_id,
                state=state,
                green_time=green_time,
                vehicle_count=vehicle_count
            )
        except Exception:
            pass

    def get_status(self) -> dict:
        """Get snapshot of current intersection state."""
        return {
            "current_green_lane": self.current_green_lane,
            "in_transition": self.in_transition,
            "transition_state": self.transition_state.value,
            "total_cycles_completed": self.total_cycles_completed,
            "emergency_active": self.emergency_lane is not None,
            "lanes": [lane.to_dict() for lane in self.lanes],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
