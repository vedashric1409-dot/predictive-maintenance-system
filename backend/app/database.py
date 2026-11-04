import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DATABASE_DIR = Path(__file__).resolve().parent.parent / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = DATABASE_DIR / "maintenance.db"


def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                machine_id TEXT,
                air_temperature REAL NOT NULL,
                process_temperature REAL NOT NULL,
                rotational_speed REAL NOT NULL,
                torque REAL NOT NULL,
                tool_wear REAL NOT NULL,
                failure_prediction INTEGER NOT NULL,
                failure_probability REAL NOT NULL,
                risk_score REAL NOT NULL,
                health_status TEXT NOT NULL,
                recommendation TEXT,
                created_at TEXT NOT NULL
            )
        """)


def save_prediction(sensor_data, prediction, machine_id=None):
    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO prediction_history (
                machine_id,
                air_temperature,
                process_temperature,
                rotational_speed,
                torque,
                tool_wear,
                failure_prediction,
                failure_probability,
                risk_score,
                health_status,
                recommendation,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            machine_id,
            sensor_data["Air temperature [K]"],
            sensor_data["Process temperature [K]"],
            sensor_data["Rotational speed [rpm]"],
            sensor_data["Torque [Nm]"],
            sensor_data["Tool wear [min]"],
            prediction["failure_prediction"],
            prediction["failure_probability"],
            prediction["risk_score"],
            prediction["health_status"],
            prediction["recommendation"],
            datetime.now(timezone.utc).isoformat()
        ))

        return cursor.lastrowid


def get_prediction_history(limit=100):
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT * FROM prediction_history
            ORDER BY id DESC
            LIMIT ?
        """, (limit,)).fetchall()

        return [dict(row) for row in rows]
