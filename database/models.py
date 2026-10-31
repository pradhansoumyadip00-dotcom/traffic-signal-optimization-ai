"""
Database Models & Table Schemas
================================
Defines database schemas for traffic management.
Fully compatible with both MySQL and SQLite.
"""

VEHICLE_LOG_SCHEMA = {
    "table_name": "vehicle_logs",
    "columns": {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "timestamp": "DATETIME NOT NULL",
        "vehicle_type": "VARCHAR(50) NOT NULL",
        "track_id": "INT NOT NULL",
        "confidence": "FLOAT DEFAULT 0.0",
        "plate_number": "VARCHAR(50) DEFAULT ''",
        "lane_id": "INT DEFAULT 0",
        "speed_estimate": "FLOAT DEFAULT 0.0",
        "snapshot_path": "VARCHAR(255) DEFAULT ''",
        "video_source": "VARCHAR(100) DEFAULT 'Video 1'"
    },
    "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_vlogs_timestamp ON vehicle_logs(timestamp)",
        "CREATE INDEX IF NOT EXISTS idx_vlogs_lane ON vehicle_logs(lane_id)",
        "CREATE INDEX IF NOT EXISTS idx_vlogs_plate ON vehicle_logs(plate_number)"
    ]
}

HELMET_VIOLATION_SCHEMA = {
    "table_name": "helmet_violations",
    "columns": {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "timestamp": "DATETIME NOT NULL",
        "confidence": "FLOAT DEFAULT 0.0",
        "plate_number": "VARCHAR(50) DEFAULT ''",
        "track_id": "INT DEFAULT 0",
        "snapshot_path": "VARCHAR(255) DEFAULT ''"
    },
    "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_helmet_timestamp ON helmet_violations(timestamp)"
    ]
}

ACCIDENT_ALERT_SCHEMA = {
    "table_name": "accident_alerts",
    "columns": {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "timestamp": "DATETIME NOT NULL",
        "confidence": "FLOAT DEFAULT 0.0",
        "severity": "VARCHAR(20) DEFAULT 'Medium'",
        "lane_id": "INT DEFAULT 0",
        "snapshot_path": "VARCHAR(255) DEFAULT ''"
    },
    "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_accident_timestamp ON accident_alerts(timestamp)"
    ]
}

SIGNAL_LOG_SCHEMA = {
    "table_name": "signal_logs",
    "columns": {
        "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
        "timestamp": "DATETIME NOT NULL",
        "lane_id": "INT NOT NULL",
        "state": "VARCHAR(10) NOT NULL",
        "green_time": "INT NOT NULL",
        "vehicle_count": "INT DEFAULT 0"
    },
    "indexes": [
        "CREATE INDEX IF NOT EXISTS idx_signal_timestamp ON signal_logs(timestamp)"
    ]
}

ALL_SCHEMAS = [
    VEHICLE_LOG_SCHEMA,
    HELMET_VIOLATION_SCHEMA,
    ACCIDENT_ALERT_SCHEMA,
    SIGNAL_LOG_SCHEMA
]
