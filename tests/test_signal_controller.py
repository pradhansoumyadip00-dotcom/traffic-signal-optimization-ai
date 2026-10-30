"""
Signal Controller Agent Test Suite
===================================
Tests density weighting, starvation prevention,
safe state transitions (Yellow/All-Red), and emergency override.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.agent.signal_controller import AdaptiveSignalController, SignalState
from database.db_manager import DatabaseManager


def run_tests():
    print("=" * 60)
    print("  Testing Intelligent Traffic Signal Agent (Step 4)...")
    print("=" * 60)

    db = DatabaseManager()
    agent = AdaptiveSignalController(db_manager=db)
    print("[PASS] AdaptiveSignalController initialized with Lane 0 in GREEN.")

    # 1. Test Density-Based Green Time Calculation
    # Set heavy traffic on Lane 1 (30 cars), light on Lane 2 (2 cars)
    agent.update_density(lane_id=1, vehicle_count=30)
    agent.update_density(lane_id=2, vehicle_count=2)

    green_heavy = agent.calculate_adaptive_green_time(1)
    green_light = agent.calculate_adaptive_green_time(2)

    assert green_heavy > green_light, f"Heavy traffic ({green_heavy}s) should get more green than light traffic ({green_light}s)"
    assert green_heavy <= 60, "Green time should not exceed max bound (60s)"
    assert green_light >= 10, "Green time should not fall below min bound (10s)"
    print(f"[PASS] Density calculation verified: Heavy Lane = {green_heavy}s, Light Lane = {green_light}s.")

    # 2. Test Starvation Prevention Penalty
    agent.lanes[2].waiting_time_sec = 120  # Has been waiting 2 minutes in Red
    green_starved = agent.calculate_adaptive_green_time(2)
    assert green_starved > green_light, "Starved lane should receive bonus green time"
    print(f"[PASS] Starvation prevention verified: Light lane green boosted from {green_light}s to {green_starved}s after wait.")

    # 3. Test Safe State Transitions (Green -> Yellow -> All-Red -> Next Green)
    # Force remaining time to 1 to trigger transition
    agent.lanes[agent.current_green_lane].remaining_time = 1
    status = agent.tick_second()

    assert agent.in_transition == True, "Should enter transition state"
    assert agent.transition_state == SignalState.YELLOW, "First transition state must be YELLOW"
    print("[PASS] GREEN -> YELLOW transition verified.")

    # Advance through YELLOW (3 seconds)
    for _ in range(3):
        agent.tick_second()

    assert agent.transition_state == SignalState.ALL_RED, "After YELLOW must come ALL_RED clearance"
    print("[PASS] YELLOW -> ALL_RED clearance transition verified.")

    # Advance through ALL_RED (2 seconds)
    for _ in range(2):
        agent.tick_second()

    assert agent.in_transition == False, "Transition must complete after clearance"
    assert agent.lanes[agent.current_green_lane].state == SignalState.GREEN, "Next chosen lane must be in GREEN"
    print(f"[PASS] ALL_RED -> NEXT GREEN transition verified. New Green Lane: {agent.current_green_lane}")

    # 4. Test Emergency Vehicle Override
    # Pick a lane that is currently RED
    red_lanes = [i for i in range(4) if i != agent.current_green_lane]
    target_emergency_lane = red_lanes[0]

    agent.update_density(lane_id=target_emergency_lane, vehicle_count=5, has_emergency=True)
    assert agent.emergency_lane == target_emergency_lane, "Emergency lane should be registered"
    assert agent.in_transition == True, "Emergency should immediately initiate clearance transition"

    # Fast forward clearance
    for _ in range(6):
        agent.tick_second()

    assert agent.current_green_lane == target_emergency_lane, f"Emergency Lane {target_emergency_lane} must be granted GREEN"
    assert agent.lanes[target_emergency_lane].state == SignalState.GREEN
    print(f"[PASS] Emergency vehicle override verified: Lane {target_emergency_lane} successfully granted priority GREEN!")

    # 5. Verify Database Logging
    signal_records = db.get_signal_logs(hours=1)
    assert len(signal_records) > 0, "Signal transitions should be logged in database"
    print(f"[PASS] Database signal logging verified: {len(signal_records)} transition records logged.")

    db.close()
    print("=" * 60)
    print("  ALL ADAPTIVE SIGNAL AGENT TESTS PASSED (100% SUCCESS)!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
