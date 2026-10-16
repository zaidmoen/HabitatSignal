"""Small SQLite store for track-level insect observations."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ObservationStore:
    """Persist one row when a new tracked detection first appears in a run."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS sightings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    track_id TEXT NOT NULL,
                    species TEXT NOT NULL,
                    first_seen TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    UNIQUE(run_id, track_id)
                )"""
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_sightings_first_seen ON sightings(first_seen)"
            )

    def record(self, run_id: str, track_id: str, species: str, confidence: float) -> None:
        with self._connect() as connection:
            connection.execute(
                """INSERT OR IGNORE INTO sightings
                   (run_id, track_id, species, first_seen, confidence)
                   VALUES (?, ?, ?, ?, ?)""",
                (run_id, track_id, species, utc_now(), round(float(confidence), 4)),
            )

    def summary(self, run_id: str | None = None) -> dict[str, Any]:
        where = "WHERE run_id = ?" if run_id else ""
        args = (run_id,) if run_id else ()
        with self._connect() as connection:
            total = connection.execute(
                f"SELECT COUNT(*) FROM sightings {where}", args
            ).fetchone()[0]
            rows = connection.execute(
                f"SELECT species, COUNT(*) AS visits FROM sightings {where} GROUP BY species ORDER BY visits DESC, species",
                args,
            ).fetchall()
            hourly = connection.execute(
                f"SELECT substr(first_seen, 12, 2) AS hour, COUNT(*) AS visits FROM sightings {where} GROUP BY hour ORDER BY hour",
                args,
            ).fetchall()
        return {
            "total_visits": total,
            "by_species": [{"species": row["species"], "visits": row["visits"]} for row in rows],
            "by_hour_utc": [{"hour": row["hour"], "visits": row["visits"]} for row in hourly],
        }

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id, track_id, species, first_seen, confidence FROM sightings ORDER BY id DESC LIMIT ?",
                (max(1, min(limit, 500)),),
            ).fetchall()
        return [dict(row) for row in rows]

