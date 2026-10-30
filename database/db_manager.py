"""
Database Manager Engine
=======================
Handles all database operations with automatic table creation,
graceful schema migrations (ALTER TABLE), dual MySQL + SQLite support,
time-based analytics aggregation, and multi-sheet Excel/CSV/PDF export.
"""

import sqlite3
import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from config.settings import USE_SQLITE, SQLITE_DB_PATH, MYSQL_CONFIG
from database.models import ALL_SCHEMAS


class DictRow(dict):
    """Dictionary subclass supporting index access and .get() for dual DB compatibility."""
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return super().__getitem__(key)


def dict_factory(cursor, row):
    fields = [col[0] for col in cursor.description]
    return DictRow(zip(fields, row))


class DatabaseManager:
    """Enterprise-grade database manager for Traffic Signal Optimization."""

    def __init__(self):
        self.use_sqlite = USE_SQLITE
        self.connection = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish database connection with graceful fallback."""
        if not self.use_sqlite:
            try:
                import mysql.connector
                self.connection = mysql.connector.connect(
                    host=MYSQL_CONFIG["host"],
                    port=MYSQL_CONFIG["port"],
                    user=MYSQL_CONFIG["user"],
                    password=MYSQL_CONFIG["password"],
                    connection_timeout=3,
                )
                cursor = self.connection.cursor()
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']}")
                cursor.execute(f"USE {MYSQL_CONFIG['database']}")
                self.connection.database = MYSQL_CONFIG["database"]
                cursor.close()
                return
            except Exception:
                self.use_sqlite = True

        db_path = Path(SQLITE_DB_PATH)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(db_path), check_same_thread=False)
        self.connection.row_factory = dict_factory

    def _get_cursor(self):
        """Returns cursor with dictionary row support across both SQLite and MySQL."""
        if self.use_sqlite:
            return self.connection.cursor()
        return self.connection.cursor(dictionary=True)

    def _create_tables(self):
        """Auto-create tables, auto-migrate new columns, and build indexes."""
        cursor = self._get_cursor()
        for schema in ALL_SCHEMAS:
            table = schema["table_name"]
            columns_def = ", ".join(
                f"{col} {dtype}" for col, dtype in schema["columns"].items()
            )
            if self.use_sqlite:
                columns_def = columns_def.replace("DATETIME", "TEXT")
                columns_def = columns_def.replace("FLOAT", "REAL")
            else:
                columns_def = columns_def.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "INT AUTO_INCREMENT PRIMARY KEY")

            cursor.execute(f"CREATE TABLE IF NOT EXISTS {table} ({columns_def})")

            # Dynamic schema auto-migration for SQLite
            if self.use_sqlite:
                cursor.execute(f"PRAGMA table_info({table})")
                existing_cols = {row['name'] if isinstance(row, dict) and 'name' in row else row[1] for row in cursor.fetchall()}
                for col, dtype in schema["columns"].items():
                    if col not in existing_cols:
                        clean_dtype = dtype.replace("DATETIME", "TEXT").replace("FLOAT", "REAL")
                        if "PRIMARY KEY" not in clean_dtype:
                            try:
                                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {clean_dtype}")
                            except Exception:
                                pass

            # Apply performance indexes
            for idx_sql in schema.get("indexes", []):
                try:
                    cursor.execute(idx_sql)
                except Exception:
                    pass
        self.connection.commit()

    # ==================== INSERT OPERATIONS ====================

    def log_vehicle(self, vehicle_type: str, track_id: int, confidence: float,
                    plate_number: str = "", lane_id: int = 0,
                    speed_estimate: float = 0.0, snapshot_path: str = ""):
        """Record a single detected vehicle into vehicle_logs."""
        cursor = self._get_cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        placeholder = "?" if self.use_sqlite else "%s"
        sql = (
            f"INSERT INTO vehicle_logs "
            f"(timestamp, vehicle_type, track_id, confidence, plate_number, lane_id, speed_estimate, snapshot_path) "
            f"VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})"
        )
        cursor.execute(sql, (now, vehicle_type, track_id, round(confidence, 3), plate_number, lane_id, round(speed_estimate, 1), snapshot_path))
        self.connection.commit()

    def log_helmet_violation(self, confidence: float, plate_number: str = "",
                              track_id: int = 0, snapshot_path: str = ""):
        """Record a detected helmet violation."""
        cursor = self._get_cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        placeholder = "?" if self.use_sqlite else "%s"
        sql = (
            f"INSERT INTO helmet_violations "
            f"(timestamp, confidence, plate_number, track_id, snapshot_path) "
            f"VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})"
        )
        cursor.execute(sql, (now, round(confidence, 3), plate_number, track_id, snapshot_path))
        self.connection.commit()

    def log_accident(self, confidence: float, severity: str = "Medium",
                     lane_id: int = 0, snapshot_path: str = ""):
        """Record an accident/collision alert."""
        cursor = self._get_cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        placeholder = "?" if self.use_sqlite else "%s"
        sql = (
            f"INSERT INTO accident_alerts "
            f"(timestamp, confidence, severity, lane_id, snapshot_path) "
            f"VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})"
        )
        cursor.execute(sql, (now, round(confidence, 3), severity, lane_id, snapshot_path))
        self.connection.commit()

    def log_signal_change(self, lane_id: int, state: str, green_time: int, vehicle_count: int = 0):
        """Record signal controller transition history."""
        cursor = self._get_cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        placeholder = "?" if self.use_sqlite else "%s"
        sql = (
            f"INSERT INTO signal_logs "
            f"(timestamp, lane_id, state, green_time, vehicle_count) "
            f"VALUES ({placeholder}, {placeholder}, {placeholder}, {placeholder}, {placeholder})"
        )
        cursor.execute(sql, (now, lane_id, state, green_time, vehicle_count))
        self.connection.commit()

    # ==================== QUERY & ANALYTICS OPERATIONS ====================

    def get_vehicles(self, hours: int = 24) -> list:
        """Get all vehicle records within the last N hours."""
        cursor = self._get_cursor()
        since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        placeholder = "?" if self.use_sqlite else "%s"
        sql = f"SELECT * FROM vehicle_logs WHERE timestamp >= {placeholder} ORDER BY timestamp DESC"
        cursor.execute(sql, (since,))
        return cursor.fetchall()

    def get_helmet_violations(self, hours: int = 24) -> list:
        """Get all helmet violations within the last N hours."""
        cursor = self._get_cursor()
        since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        placeholder = "?" if self.use_sqlite else "%s"
        sql = f"SELECT * FROM helmet_violations WHERE timestamp >= {placeholder} ORDER BY timestamp DESC"
        cursor.execute(sql, (since,))
        return cursor.fetchall()

    def get_accidents(self, hours: int = 24) -> list:
        """Get all accident alerts within the last N hours."""
        cursor = self._get_cursor()
        since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        placeholder = "?" if self.use_sqlite else "%s"
        sql = f"SELECT * FROM accident_alerts WHERE timestamp >= {placeholder} ORDER BY timestamp DESC"
        cursor.execute(sql, (since,))
        return cursor.fetchall()

    def get_signal_logs(self, hours: int = 24) -> list:
        """Get signal transition history within the last N hours."""
        cursor = self._get_cursor()
        since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        placeholder = "?" if self.use_sqlite else "%s"
        sql = f"SELECT * FROM signal_logs WHERE timestamp >= {placeholder} ORDER BY timestamp DESC"
        cursor.execute(sql, (since,))
        return cursor.fetchall()

    def get_vehicle_count_by_type(self, hours: int = 24) -> list:
        """Get vehicle distribution grouped by vehicle type (for pie charts)."""
        cursor = self._get_cursor()
        since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        placeholder = "?" if self.use_sqlite else "%s"
        sql = (
            f"SELECT vehicle_type, COUNT(*) as count FROM vehicle_logs "
            f"WHERE timestamp >= {placeholder} "
            f"GROUP BY vehicle_type ORDER BY count DESC"
        )
        cursor.execute(sql, (since,))
        return cursor.fetchall()

    def get_lane_distribution(self, hours: int = 24) -> list:
        """Get vehicle distribution per lane (Lane 0, 1, 2, 3) for the Agent."""
        cursor = self._get_cursor()
        since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        placeholder = "?" if self.use_sqlite else "%s"
        sql = (
            f"SELECT lane_id, COUNT(*) as count FROM vehicle_logs "
            f"WHERE timestamp >= {placeholder} "
            f"GROUP BY lane_id ORDER BY lane_id ASC"
        )
        cursor.execute(sql, (since,))
        return cursor.fetchall()

    def get_hourly_traffic(self, hours: int = 24) -> list:
        """Get hourly vehicle counts for bar and line charts."""
        cursor = self._get_cursor()
        since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")
        placeholder = "?" if self.use_sqlite else "%s"
        if self.use_sqlite:
            sql = (
                f"SELECT strftime('%H', timestamp) as hour, COUNT(*) as count FROM vehicle_logs "
                f"WHERE timestamp >= {placeholder} "
                f"GROUP BY hour ORDER BY hour ASC"
            )
        else:
            sql = (
                f"SELECT HOUR(timestamp) as hour, COUNT(*) as count FROM vehicle_logs "
                f"WHERE timestamp >= {placeholder} "
                f"GROUP BY hour ORDER BY hour ASC"
            )
        cursor.execute(sql, (since,))
        return cursor.fetchall()

    def get_summary_stats(self, hours: int = 24) -> dict:
        """Returns high-level analytical summary dictionary in one single call."""
        vehicles = self.get_vehicles(hours=hours)
        violations = self.get_helmet_violations(hours=hours)
        accidents = self.get_accidents(hours=hours)
        by_type = self.get_vehicle_count_by_type(hours=hours)
        by_lane = self.get_lane_distribution(hours=hours)

        total_v = len(vehicles)

        return {
            "total_vehicles": total_v,
            "total_violations": len(violations),
            "total_accidents": len(accidents),
            "vehicle_types": {row["vehicle_type"]: row["count"] for row in by_type},
            "lane_counts": {row["lane_id"]: row["count"] for row in by_lane},
            "hours_queried": hours,
            "query_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    # ==================== EXPORT ENGINES ====================

    def export_to_excel(self, output_path: str, hours: int = 24) -> str:
        """Export multi-sheet professional Excel report."""
        v_data = [dict(r) for r in self.get_vehicles(hours=hours)]
        h_data = [dict(r) for r in self.get_helmet_violations(hours=hours)]
        a_data = [dict(r) for r in self.get_accidents(hours=hours)]
        s_data = [dict(r) for r in self.get_signal_logs(hours=hours)]

        df_v = pd.DataFrame(v_data) if v_data else pd.DataFrame(columns=["id", "timestamp", "vehicle_type", "track_id", "confidence", "lane_id", "speed_estimate"])
        df_h = pd.DataFrame(h_data) if h_data else pd.DataFrame(columns=["id", "timestamp", "confidence", "track_id"])
        df_a = pd.DataFrame(a_data) if a_data else pd.DataFrame(columns=["id", "timestamp", "severity", "confidence"])
        df_s = pd.DataFrame(s_data) if s_data else pd.DataFrame(columns=["id", "timestamp", "lane_id", "state", "green_time"])

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_v.to_excel(writer, sheet_name="Vehicle Logs", index=False)
            df_h.to_excel(writer, sheet_name="Helmet Violations", index=False)
            df_a.to_excel(writer, sheet_name="Accident Alerts", index=False)
            df_s.to_excel(writer, sheet_name="Signal History", index=False)

        return output_path

    def export_to_csv(self, output_path: str, hours: int = 24) -> str:
        """Export main vehicle logs to a clean CSV file."""
        v_data = [dict(r) for r in self.get_vehicles(hours=hours)]
        df = pd.DataFrame(v_data) if v_data else pd.DataFrame(columns=["id", "timestamp", "vehicle_type", "track_id", "confidence", "lane_id", "speed_estimate"])
        df.to_csv(output_path, index=False)
        return output_path

    def export_to_pdf(self, output_path: str, hours: int = 24) -> str:
        """Generate clean, executive PDF audit report using fpdf2."""
        from fpdf import FPDF
        from fpdf.enums import XPos, YPos

        vehicles = [dict(r) for r in self.get_vehicles(hours=hours)]
        violations = [dict(r) for r in self.get_helmet_violations(hours=hours)]
        accidents = [dict(r) for r in self.get_accidents(hours=hours)]

        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        # Header Title
        pdf.set_font("helvetica", "B", 18)
        pdf.set_text_color(21, 27, 38)
        pdf.cell(0, 10, "SMART TRAFFIC MANAGEMENT SYSTEM", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

        pdf.set_font("helvetica", "", 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, f"Executive Audit Report | Window: Last {hours} Hours | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        pdf.ln(5)

        # Summary Metrics Box
        pdf.set_fill_color(34, 43, 58)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", "B", 11)
        pdf.cell(0, 8, "  1. INCIDENT & TRAFFIC SUMMARY", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)

        pdf.set_text_color(30, 41, 59)
        pdf.set_font("helvetica", "", 10)
        pdf.ln(2)
        pdf.cell(63, 7, f"Total Vehicles: {len(vehicles)}", border=1)
        pdf.cell(63, 7, f"Helmet Infractions: {len(violations)}", border=1)
        pdf.cell(63, 7, f"Accident Alerts: {len(accidents)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, border=1)
        pdf.ln(6)

        # Table: Recent Vehicle Transits
        pdf.set_fill_color(34, 43, 58)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("helvetica", "B", 11)
        pdf.cell(0, 8, "  2. RECENT VEHICLE TRANSIT AUDIT (TOP 25)", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)

        pdf.set_fill_color(241, 245, 249)
        pdf.set_text_color(15, 23, 42)
        pdf.set_font("helvetica", "B", 9)
        pdf.ln(2)
        pdf.cell(45, 7, "Timestamp", border=1, fill=True)
        pdf.cell(35, 7, "Vehicle Class", border=1, fill=True)
        pdf.cell(35, 7, "Track ID", border=1, fill=True)
        pdf.cell(40, 7, "Corridor Lane", border=1, fill=True)
        pdf.cell(35, 7, "Speed (km/h)", new_x=XPos.LMARGIN, new_y=YPos.NEXT, border=1, fill=True)

        pdf.set_font("helvetica", "", 9)
        for row in vehicles[:25]:
            pdf.cell(45, 6, str(row.get("timestamp", ""))[:19], border=1)
            pdf.cell(35, 6, str(row.get("vehicle_type", "Unknown")), border=1)
            pdf.cell(35, 6, f"#{row.get('track_id', 0)}", border=1)
            lane_num = int(row.get('lane_id') or 0) + 1
            pdf.cell(40, 6, f"Corridor {lane_num}", border=1)
            speed_val = float(row.get('speed_estimate') or 0.0)
            pdf.cell(35, 6, f"{speed_val:.1f} km/h", new_x=XPos.LMARGIN, new_y=YPos.NEXT, border=1)

        # Section 3: Helmet Safety Infractions
        if violations:
            pdf.ln(4)
            pdf.set_fill_color(34, 43, 58)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("helvetica", "B", 11)
            pdf.cell(0, 8, "  3. HELMET SAFETY INFRACTIONS", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)

            pdf.set_fill_color(241, 245, 249)
            pdf.set_text_color(15, 23, 42)
            pdf.set_font("helvetica", "B", 9)
            pdf.ln(2)
            pdf.cell(55, 7, "Timestamp", border=1, fill=True)
            pdf.cell(45, 7, "Infraction", border=1, fill=True)
            pdf.cell(45, 7, "Confidence", border=1, fill=True)
            pdf.cell(45, 7, "Track ID", new_x=XPos.LMARGIN, new_y=YPos.NEXT, border=1, fill=True)

            pdf.set_font("helvetica", "", 9)
            for row in violations[:15]:
                pdf.cell(55, 6, str(row.get("timestamp", ""))[:19], border=1)
                pdf.cell(45, 6, "No Helmet", border=1)
                conf_val = float(row.get('confidence') or 0.0)
                pdf.cell(45, 6, f"{conf_val:.2f}", border=1)
                pdf.cell(45, 6, str(row.get("track_id", 0)), new_x=XPos.LMARGIN, new_y=YPos.NEXT, border=1)

        # Section 4: Incident and Accident Alerts
        if accidents:
            pdf.ln(4)
            pdf.set_fill_color(34, 43, 58)
            pdf.set_text_color(255, 255, 255)
            pdf.set_font("helvetica", "B", 11)
            pdf.cell(0, 8, "  4. CRITICAL INCIDENT AND ACCIDENT ALERTS", new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)

            pdf.set_fill_color(241, 245, 249)
            pdf.set_text_color(15, 23, 42)
            pdf.set_font("helvetica", "B", 9)
            pdf.ln(2)
            pdf.cell(50, 7, "Timestamp", border=1, fill=True)
            pdf.cell(45, 7, "Severity", border=1, fill=True)
            lane_num = int(row.get('lane_id') or 0) + 1
            pdf.cell(45, 7, f"Corridor {lane_num}", border=1)
            conf_val = float(row.get('confidence') or 0.0)
            pdf.cell(50, 7, f"{conf_val:.2f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, border=1)

            pdf.set_font("helvetica", "", 9)
            for row in accidents[:10]:
                pdf.cell(50, 6, str(row.get("timestamp", ""))[:19], border=1)
                pdf.cell(45, 6, str(row.get("severity", "Medium")), border=1)
                lane_num = int(row.get('lane_id') or 0) + 1
                pdf.cell(45, 6, f"Corridor {lane_num}", border=1)
                conf_val = float(row.get('confidence') or 0.0)
                pdf.cell(50, 6, f"{conf_val:.2f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, border=1)

        pdf.output(output_path)
        return output_path

    def close(self):
        """Close active database connection."""
        if self.connection:
            try:
                self.connection.close()
            except Exception:
                pass
