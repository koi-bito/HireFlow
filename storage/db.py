"""
storage/db.py
-------------
Hand-written SQLite database layer for runs persistence and recruiter feedback.
No ORM dependencies; clean parameterized SQL with indexing and foreign-key constraints.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional, Union

import config

logger = logging.getLogger(__name__)


def get_connection(db_path: Optional[Union[str, Path]] = None) -> sqlite3.Connection:
    """Returns a SQLite connection with row_factory enabled and foreign keys enforced."""
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(db_path: Optional[Union[str, Path]] = None) -> None:
    """Initializes tables and indexes using hand-written DDL."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                job_title TEXT,
                raw_jd TEXT,
                candidates_csv_path TEXT,
                result_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                candidate_email TEXT NOT NULL,
                job_title TEXT NOT NULL,
                action TEXT NOT NULL CHECK(action IN ('shortlisted', 'rejected', 'hired')),
                candidate_summary TEXT,
                candidate_skills TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE SET NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_candidate_email ON feedback(candidate_email);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_job_title ON feedback(job_title);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_action ON feedback(action);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_run_id ON feedback(run_id);")
        conn.commit()
        logger.info("Database initialized successfully at %s", db_path or config.DB_PATH)
    finally:
        conn.close()


def save_run(
    run_id: str,
    result: dict[str, Any],
    job_title: str = "",
    raw_jd: str = "",
    candidates_csv_path: str = "",
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Saves a completed pipeline run to SQLite."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO runs (run_id, job_title, raw_jd, candidates_csv_path, result_json)
            VALUES (?, ?, ?, ?, ?);
            """,
            (
                run_id,
                job_title or (result.get("parsed_jd") or {}).get("job_title", "Unknown"),
                raw_jd,
                candidates_csv_path,
                json.dumps(result),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_run(run_id: str, db_path: Optional[Union[str, Path]] = None) -> Optional[dict[str, Any]]:
    """Retrieves a run result by run_id from SQLite."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT result_json FROM runs WHERE run_id = ?;", (run_id,))
        row = cursor.fetchone()
        if row:
            return json.loads(row["result_json"])
        return None
    finally:
        conn.close()


def list_runs(
    limit: int = 10,
    offset: int = 0,
    db_path: Optional[Union[str, Path]] = None,
) -> tuple[list[dict[str, Any]], int]:
    """Lists past runs with pagination, returning (items, total_count)."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) AS total FROM runs;")
        total = cursor.fetchone()["total"]

        cursor.execute(
            """
            SELECT run_id, job_title, created_at, result_json
            FROM runs
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?;
            """,
            (limit, offset),
        )
        rows = cursor.fetchall()
        items = []
        for r in rows:
            data = json.loads(r["result_json"])
            items.append({
                "run_id": r["run_id"],
                "job_title": r["job_title"],
                "created_at": r["created_at"],
                "ranked_candidates_count": len(data.get("ranked_candidates", [])),
                "errors_count": len(data.get("errors", [])),
            })
        return items, total
    finally:
        conn.close()


def record_feedback(
    candidate_email: str,
    job_title: str,
    action: str,
    run_id: Optional[str] = None,
    candidate_summary: str = "",
    candidate_skills: str = "",
    db_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Records recruiter feedback for a candidate on a job role."""
    action_clean = action.lower().strip()
    if action_clean not in {"shortlisted", "rejected", "hired"}:
        raise ValueError(f"Invalid feedback action: '{action}'. Must be 'shortlisted', 'rejected', or 'hired'.")

    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO feedback (run_id, candidate_email, job_title, action, candidate_summary, candidate_skills)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            (
                run_id,
                candidate_email.strip().lower(),
                job_title.strip(),
                action_clean,
                candidate_summary,
                candidate_skills,
            ),
        )
        feedback_id = cursor.lastrowid
        conn.commit()

        cursor.execute("SELECT * FROM feedback WHERE id = ?;", (feedback_id,))
        row = dict(cursor.fetchone())
        return row
    finally:
        conn.close()


def get_feedback_for_role(
    job_title: str,
    db_path: Optional[Union[str, Path]] = None,
) -> list[dict[str, Any]]:
    """Retrieves past feedback for a specific role or role pattern."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        pattern = f"%{job_title.strip().lower()}%"
        cursor.execute(
            """
            SELECT * FROM feedback
            WHERE LOWER(job_title) LIKE ?
            ORDER BY created_at DESC;
            """,
            (pattern,),
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def get_all_feedback(
    db_path: Optional[Union[str, Path]] = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Retrieves all feedback records."""
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM feedback ORDER BY created_at DESC LIMIT ?;", (limit,))
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()
