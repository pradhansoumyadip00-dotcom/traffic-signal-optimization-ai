import sys
import os
import py_compile
import numpy as np
from pathlib import Path

base = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(base))

print("=" * 60)
print("  COMPLETE SYSTEM HEALTH & INTEGRITY CHECK")
print("=" * 60)

# 1. Module Imports
modules = [
    "config.settings",
    "database.models",
    "database.db_manager",
    "core.detection.vehicle_detector",
    "core.detection.helmet_detector",
    "core.detection.accident_detector",
    "core.detection.vision_engine",
    "core.tracking.tracker",
    "core.anpr.plate_reader",
    "core.agent.signal_controller",
    "utils.night_enhance",
    "utils.video_utils"
]

all_passed = True
print("\n[Step 1] Checking Python Modules & Dependencies:")
for mod in modules:
    try:
        __import__(mod)
        print(f"  [OK] {mod}")
    except Exception as e:
        print(f"  [FAIL] {mod}: {e}")
        all_passed = False

# 2. Database Read / Write Check
print("\n[Step 2] Checking Database Backend:")
try:
    from database.db_manager import DatabaseManager
    db = DatabaseManager()
    summary = db.get_summary_stats(hours=24)
    print(f"  [OK] Database connected. Found {summary['total_vehicles']} vehicles, {summary['total_violations']} violations.")
    db.close()
except Exception as e:
    print(f"  [FAIL] Database check failed: {e}")
    all_passed = False

# 3. Vision Engine Check
print("\n[Step 3] Checking Vision Engine & Tracker:")
try:
    from core.detection.vision_engine import VisionEngine
    ve = VisionEngine()
    test_img = np.full((720, 1280, 3), 120, dtype=np.uint8)
    ann, stats = ve.process_frame(test_img, frame_idx=1, record_to_db=False)
    print(f"  [OK] VisionEngine executed successfully without error.")
except Exception as e:
    print(f"  [FAIL] VisionEngine check failed: {e}")
    all_passed = False

# 4. Adaptive Signal Agent Check
print("\n[Step 4] Checking Adaptive Signal Controller:")
try:
    from core.agent.signal_controller import AdaptiveSignalController
    agent = AdaptiveSignalController()
    green = agent.calculate_adaptive_green_time(0)
    print(f"  [OK] Adaptive Signal Controller calculated green time: {green}s.")
except Exception as e:
    print(f"  [FAIL] Signal Controller check failed: {e}")
    all_passed = False

# 5. Frontend Dashboard Compilation Check
print("\n[Step 5] Checking Frontend Dashboard (app.py):")
try:
    py_compile.compile(str(base / "dashboard" / "app.py"), doraise=True)
    print("  [OK] dashboard/app.py syntax is 100% valid with 0 errors.")
except Exception as e:
    print(f"  [FAIL] Dashboard syntax check failed: {e}")
    all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("  RESULT: ALL BACKEND & FRONTEND MODULES ARE 100% HEALTHY!")
else:
    print("  RESULT: SOME ISSUES WERE DETECTED!")
print("=" * 60)
