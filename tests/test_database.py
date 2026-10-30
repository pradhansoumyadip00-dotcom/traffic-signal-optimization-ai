"""
Database Engine Verification Test Suite
========================================
Runs full validation on table creation, data insertion,
time filtering, aggregation, and Excel/CSV exporting.
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from database.db_manager import DatabaseManager


def run_tests():
    print("=" * 60)
    print("  Running Database Engine Test Suite...")
    print("=" * 60)

    db = DatabaseManager()
    print("[PASS] Database connection and table verification successful.")

    # 1. Test inserting records
    db.log_vehicle(
        vehicle_type="Car",
        track_id=101,
        confidence=0.96,
        plate_number="WB02AB1234",
        lane_id=1,
        speed_estimate=45.5
    )
    db.log_vehicle(
        vehicle_type="Motorcycle",
        track_id=102,
        confidence=0.91,
        plate_number="WB24X5678",
        lane_id=2,
        speed_estimate=38.0
    )
    db.log_vehicle(
        vehicle_type="Bus",
        track_id=103,
        confidence=0.88,
        plate_number="WB01C9999",
        lane_id=1,
        speed_estimate=30.0
    )
    print("[PASS] Vehicle insertion tested successfully.")

    # 2. Test Helmet Violation
    db.log_helmet_violation(confidence=0.93, plate_number="WB24X5678", track_id=102)
    print("[PASS] Helmet violation insertion tested successfully.")

    # 3. Test Accident Alert
    db.log_accident(confidence=0.89, severity="High", lane_id=1)
    print("[PASS] Accident alert insertion tested successfully.")

    # 4. Test Signal Change Log
    db.log_signal_change(lane_id=1, state="GREEN", green_time=45, vehicle_count=12)
    print("[PASS] Signal change insertion tested successfully.")

    # 5. Test Queries & Filters
    v_1h = db.get_vehicles(hours=1)
    assert len(v_1h) >= 3, "Vehicle query failed"

    h_1h = db.get_helmet_violations(hours=1)
    assert len(h_1h) >= 1, "Helmet violation query failed"

    a_1h = db.get_accidents(hours=1)
    assert len(a_1h) >= 1, "Accident alert query failed"

    s_1h = db.get_signal_logs(hours=1)
    assert len(s_1h) >= 1, "Signal log query failed"
    print("[PASS] Time-based queries (get_vehicles, get_violations, get_accidents, get_signals) verified.")

    # 6. Test Summary Stats & Aggregations
    summary = db.get_summary_stats(hours=24)
    assert summary["total_vehicles"] >= 3, "Summary total_vehicles failed"
    assert "vehicle_types" in summary, "Summary vehicle_types missing"
    assert "lane_counts" in summary, "Summary lane_counts missing"
    print(f"[PASS] Summary stats API verified: {summary['total_vehicles']} total vehicles, {summary['total_violations']} violations.")

    # 7. Test Lane Distribution (Critical for Agent)
    lanes = db.get_lane_distribution(hours=24)
    assert len(lanes) > 0, "Lane distribution query failed"
    print(f"[PASS] Lane distribution query for Adaptive Agent verified: {[dict(l) for l in lanes]}")

    # 8. Test Export Engines
    export_dir = Path(__file__).resolve().parent.parent / "data"
    excel_path = str(export_dir / "test_report.xlsx")
    csv_path = str(export_dir / "test_report.csv")
    pdf_path = str(export_dir / "test_report.pdf")

    db.export_to_excel(excel_path, hours=24)
    assert Path(excel_path).exists(), "Excel export file not created"
    print(f"[PASS] Multi-sheet Excel report exported successfully: {excel_path}")

    db.export_to_csv(csv_path, hours=24)
    assert Path(csv_path).exists(), "CSV export file not created"
    print(f"[PASS] CSV report exported successfully: {csv_path}")

    db.export_to_pdf(pdf_path, hours=24)
    assert Path(pdf_path).exists(), "PDF export file not created"
    print(f"[PASS] PDF report exported successfully: {pdf_path}")

    db.close()
    print("=" * 60)
    print("  ALL DATABASE ENGINE TESTS PASSED (100% SUCCESS)!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
