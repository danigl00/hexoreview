"""SQLite store for reviewers, review status and seizure marks.

One database holds every clinician's work, but every read is filtered by
clinician_id, so a reviewer only ever sees their own marks. The database holds
blinded labels only; patient identity lives in the separate blinding map that
clinicians never have access to.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS clinicians (
    clinician_id TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    passcode     TEXT,
    created_at   TEXT NOT NULL,
    role         TEXT NOT NULL DEFAULT 'reviewer'   -- 'reviewer' | 'coordinator'
);

CREATE TABLE IF NOT EXISTS recordings (
    blind_label TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    duration_s  REAL,
    sfreq       REAL,
    added_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    clinician_id   TEXT NOT NULL,
    blind_label    TEXT NOT NULL,
    status         TEXT NOT NULL,        -- 'in_progress' | 'reviewed'
    opened_at      TEXT,
    completed_at   TEXT,
    active_seconds REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (clinician_id, blind_label)
);

-- one row per sitting, so a night reviewed over several sessions can be broken
-- down as well as totalled
CREATE TABLE IF NOT EXISTS review_sessions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    clinician_id   TEXT NOT NULL,
    blind_label    TEXT NOT NULL,
    started_at     TEXT NOT NULL,
    last_active_at TEXT,
    active_seconds REAL NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sessions
    ON review_sessions (clinician_id, blind_label);

CREATE TABLE IF NOT EXISTS annotations (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    clinician_id TEXT NOT NULL,
    blind_label  TEXT NOT NULL,
    onset_s      REAL NOT NULL,
    offset_s     REAL NOT NULL,
    note         TEXT DEFAULT '',
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ann ON annotations (clinician_id, blind_label);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Store:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self):
        """Add columns introduced after a database was first created."""
        cols = {
            r["name"] for r in self.conn.execute("PRAGMA table_info(reviews)")
        }
        if "active_seconds" not in cols:
            self.conn.execute(
                "ALTER TABLE reviews ADD COLUMN active_seconds REAL NOT NULL DEFAULT 0"
            )
        cols = {
            r["name"] for r in self.conn.execute("PRAGMA table_info(clinicians)")
        }
        if "role" not in cols:
            self.conn.execute(
                "ALTER TABLE clinicians ADD COLUMN role TEXT NOT NULL DEFAULT 'reviewer'"
            )

    # -- clinicians ----------------------------------------------------------
    def add_clinician(
        self, clinician_id: str, display_name: str, passcode: str = "",
        role: str = "reviewer",
    ):
        self.conn.execute(
            "INSERT OR REPLACE INTO clinicians VALUES (?,?,?,?,?)",
            (clinician_id, display_name, passcode, _now(), role),
        )
        self.conn.commit()

    def role_of(self, clinician_id: str) -> str:
        row = self.conn.execute(
            "SELECT role FROM clinicians WHERE clinician_id = ?", (clinician_id,)
        ).fetchone()
        return (row["role"] if row else "reviewer") or "reviewer"

    def reviewers(self) -> list[sqlite3.Row]:
        """Everyone who actually reviews recordings, excluding coordinators."""
        return self.conn.execute(
            "SELECT * FROM clinicians WHERE role != 'coordinator' ORDER BY clinician_id"
        ).fetchall()

    def clinicians(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM clinicians ORDER BY clinician_id"
        ).fetchall()

    def check_login(self, clinician_id: str, passcode: str) -> bool:
        row = self.conn.execute(
            "SELECT passcode FROM clinicians WHERE clinician_id = ?", (clinician_id,)
        ).fetchone()
        if row is None:
            return False
        return (row["passcode"] or "") == (passcode or "")

    # -- recordings ----------------------------------------------------------
    def register_recording(
        self, blind_label: str, filename: str, duration_s=None, sfreq=None
    ):
        self.conn.execute(
            """INSERT INTO recordings (blind_label, filename, duration_s, sfreq, added_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(blind_label) DO UPDATE SET
                 filename=excluded.filename,
                 duration_s=COALESCE(excluded.duration_s, recordings.duration_s),
                 sfreq=COALESCE(excluded.sfreq, recordings.sfreq)""",
            (blind_label, filename, duration_s, sfreq, _now()),
        )
        self.conn.commit()

    def register_dir(self, blinded_dir) -> int:
        """Register every blinded EDF found in a folder. Returns the count."""
        paths = sorted(Path(blinded_dir).glob("*.edf"))
        for path in paths:
            self.register_recording(path.stem, path.name)
        return len(paths)

    def worklist(self, clinician_id: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT r.blind_label,
                      r.duration_s,
                      COALESCE(v.status, 'not_started') AS status,
                      (SELECT COUNT(*) FROM annotations a
                        WHERE a.blind_label = r.blind_label
                          AND a.clinician_id = ?) AS n_marks
                 FROM recordings r
                 LEFT JOIN reviews v
                   ON v.blind_label = r.blind_label AND v.clinician_id = ?
                ORDER BY r.blind_label""",
            (clinician_id, clinician_id),
        ).fetchall()
        return [dict(r) for r in rows]

    # -- review status -------------------------------------------------------
    def open_recording(self, clinician_id: str, blind_label: str):
        self.conn.execute(
            """INSERT INTO reviews (clinician_id, blind_label, status, opened_at)
               VALUES (?,?,'in_progress',?)
               ON CONFLICT(clinician_id, blind_label) DO NOTHING""",
            (clinician_id, blind_label, _now()),
        )
        self.conn.commit()

    def set_status(self, clinician_id: str, blind_label: str, status: str):
        completed = _now() if status == "reviewed" else None
        self.conn.execute(
            """INSERT INTO reviews (clinician_id, blind_label, status, opened_at, completed_at)
               VALUES (?,?,?,?,?)
               ON CONFLICT(clinician_id, blind_label) DO UPDATE SET
                 status=excluded.status, completed_at=excluded.completed_at""",
            (clinician_id, blind_label, status, _now(), completed),
        )
        self.conn.commit()

    def start_session(self, clinician_id: str, blind_label: str) -> int:
        """Begin a sitting on a recording. Returns the session id."""
        cur = self.conn.execute(
            """INSERT INTO review_sessions
               (clinician_id, blind_label, started_at, last_active_at, active_seconds)
               VALUES (?,?,?,?,0)""",
            (clinician_id, blind_label, _now(), _now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def log_activity(
        self, session_id: int, clinician_id: str, blind_label: str, seconds: float
    ):
        """Add active review time to both the sitting and the running total."""
        if seconds <= 0:
            return
        self.conn.execute(
            """UPDATE review_sessions
                  SET active_seconds = active_seconds + ?, last_active_at = ?
                WHERE id = ?""",
            (float(seconds), _now(), session_id),
        )
        self.conn.execute(
            """UPDATE reviews
                  SET active_seconds = active_seconds + ?
                WHERE clinician_id = ? AND blind_label = ?""",
            (float(seconds), clinician_id, blind_label),
        )
        self.conn.commit()

    def active_seconds(self, clinician_id: str, blind_label: str) -> float:
        row = self.conn.execute(
            "SELECT active_seconds FROM reviews WHERE clinician_id=? AND blind_label=?",
            (clinician_id, blind_label),
        ).fetchone()
        return float(row["active_seconds"]) if row else 0.0

    def session_counts(self) -> dict[tuple[str, str], int]:
        rows = self.conn.execute(
            """SELECT clinician_id, blind_label, COUNT(*) AS n
                 FROM review_sessions
                WHERE active_seconds > 0
                GROUP BY clinician_id, blind_label"""
        ).fetchall()
        return {(r["clinician_id"], r["blind_label"]): int(r["n"]) for r in rows}

    def status_of(self, clinician_id: str, blind_label: str) -> str:
        row = self.conn.execute(
            "SELECT status FROM reviews WHERE clinician_id=? AND blind_label=?",
            (clinician_id, blind_label),
        ).fetchone()
        return row["status"] if row else "not_started"

    # -- annotations ---------------------------------------------------------
    def add_annotation(
        self, clinician_id: str, blind_label: str, onset_s: float, offset_s: float,
        note: str = "",
    ) -> int:
        onset_s, offset_s = sorted((float(onset_s), float(offset_s)))
        cur = self.conn.execute(
            """INSERT INTO annotations
               (clinician_id, blind_label, onset_s, offset_s, note, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?)""",
            (clinician_id, blind_label, onset_s, offset_s, note, _now(), _now()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def update_annotation(self, ann_id: int, onset_s=None, offset_s=None, note=None):
        row = self.conn.execute(
            "SELECT * FROM annotations WHERE id=?", (ann_id,)
        ).fetchone()
        if row is None:
            return
        onset = float(row["onset_s"] if onset_s is None else onset_s)
        offset = float(row["offset_s"] if offset_s is None else offset_s)
        onset, offset = sorted((onset, offset))
        self.conn.execute(
            "UPDATE annotations SET onset_s=?, offset_s=?, note=?, updated_at=? WHERE id=?",
            (onset, offset, row["note"] if note is None else note, _now(), ann_id),
        )
        self.conn.commit()

    def delete_annotation(self, ann_id: int):
        self.conn.execute("DELETE FROM annotations WHERE id=?", (ann_id,))
        self.conn.commit()

    def annotations(self, clinician_id: str, blind_label: str) -> list[dict]:
        rows = self.conn.execute(
            """SELECT * FROM annotations
                WHERE clinician_id=? AND blind_label=?
                ORDER BY onset_s""",
            (clinician_id, blind_label),
        ).fetchall()
        return [dict(r) for r in rows]

    def all_annotations(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM annotations ORDER BY clinician_id, blind_label, onset_s"
        ).fetchall()
        return [dict(r) for r in rows]

    def all_reviews(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM reviews ORDER BY clinician_id, blind_label"
        ).fetchall()
        return [dict(r) for r in rows]