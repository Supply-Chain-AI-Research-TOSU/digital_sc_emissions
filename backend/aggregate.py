import sqlite3
from datetime import datetime
from typing import Optional
import threading


DB_PATH = "./aggregate.db"
_lock = threading.Lock()


def _init_db():
    with _lock:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                method TEXT,
                duration_s REAL,
                emissions_kg REAL,
                energy_kwh REAL,
                tokens_in INTEGER DEFAULT 0,
                tokens_out INTEGER DEFAULT 0
            )
        """)
        conn.commit()
        conn.close()


_init_db()


def record_request(
    method: str,
    duration_s: float,
    emissions_kg: float,
    energy_kwh: float,
    tokens_in: int = 0,
    tokens_out: int = 0,
) -> None:
    timestamp = datetime.utcnow().isoformat()
    with _lock:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO session_log (timestamp, method, duration_s, emissions_kg, energy_kwh, tokens_in, tokens_out) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                timestamp,
                method,
                duration_s,
                emissions_kg,
                energy_kwh,
                tokens_in,
                tokens_out,
            ),
        )
        conn.commit()
        conn.close()


def get_aggregate() -> dict:
    with _lock:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM session_log")
        n_requests = cursor.fetchone()[0]

        if n_requests == 0:
            conn.close()
            return {
                "n_requests": 0,
                "total_emissions_g": 0.0,
                "total_energy_kwh": 0.0,
                "avg_emissions_g_per_request": 0.0,
                "method_counts": {},
                "session_start": "",
                "last_updated": "",
            }

        cursor.execute("SELECT SUM(emissions_kg), SUM(energy_kwh) FROM session_log")
        total_emissions_kg, total_energy_kwh = cursor.fetchone()

        cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM session_log")
        session_start, last_updated = cursor.fetchone()

        cursor.execute("SELECT method, COUNT(*) FROM session_log GROUP BY method")
        method_rows = cursor.fetchall()
        method_counts = {row[0]: row[1] for row in method_rows}

        conn.close()

        total_emissions_g = (total_emissions_kg or 0) * 1000
        avg_emissions_g_per_request = (
            total_emissions_g / n_requests if n_requests > 0 else 0.0
        )

        return {
            "n_requests": n_requests,
            "total_emissions_g": total_emissions_g,
            "total_energy_kwh": total_energy_kwh or 0.0,
            "avg_emissions_g_per_request": avg_emissions_g_per_request,
            "method_counts": method_counts,
            "session_start": session_start,
            "last_updated": last_updated,
        }
