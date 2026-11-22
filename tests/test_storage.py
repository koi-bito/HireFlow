"""
tests/test_storage.py
----------------------
Tests for SQLite schema, hand-written SQL queries, feedback recording, and runs persistence.
"""

import tempfile
from pathlib import Path

import pytest

from storage.db import (
    get_feedback_for_role,
    get_run,
    init_db,
    list_runs,
    record_feedback,
    save_run,
)


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)
    init_db(db_path)
    yield db_path
    try:
        if db_path.exists():
            db_path.unlink()
    except Exception:
        pass


def test_schema_initialization(temp_db):
    # Running init_db twice should be idempotent
    init_db(temp_db)
    runs, total = list_runs(db_path=temp_db)
    assert total == 0
    assert runs == []


def test_save_and_get_run(temp_db):
    run_id = "test-run-123"
    result_data = {
        "run_id": run_id,
        "parsed_jd": {"job_title": "ML Engineer"},
        "ranked_candidates": [{"name": "Alice", "score": 90}],
        "errors": [],
    }

    save_run(run_id, result_data, job_title="ML Engineer", db_path=temp_db)

    # Survives retrieval
    retrieved = get_run(run_id, db_path=temp_db)
    assert retrieved is not None
    assert retrieved["run_id"] == run_id
    assert retrieved["ranked_candidates"][0]["name"] == "Alice"

    # List runs pagination
    items, total = list_runs(limit=5, offset=0, db_path=temp_db)
    assert total == 1
    assert items[0]["run_id"] == run_id
    assert items[0]["job_title"] == "ML Engineer"


def test_record_feedback_validations(temp_db):
    # Successful record
    fb = record_feedback(
        candidate_email="alice@example.com",
        job_title="Senior ML Engineer",
        action="shortlisted",
        db_path=temp_db,
    )
    assert fb["candidate_email"] == "alice@example.com"
    assert fb["action"] == "shortlisted"

    # Invalid action raises ValueError
    with pytest.raises(ValueError):
        record_feedback(
            candidate_email="bob@example.com",
            job_title="ML Engineer",
            action="promoted",  # invalid action
            db_path=temp_db,
        )


def test_get_feedback_for_role(temp_db):
    record_feedback("alice@test.com", "Senior ML Engineer", "hired", db_path=temp_db)
    record_feedback("bob@test.com", "Senior ML Engineer", "rejected", db_path=temp_db)
    record_feedback("carol@test.com", "Frontend Developer", "shortlisted", db_path=temp_db)

    ml_feedback = get_feedback_for_role("ML Engineer", db_path=temp_db)
    assert len(ml_feedback) == 2
    emails = [f["candidate_email"] for f in ml_feedback]
    assert "alice@test.com" in emails
    assert "bob@test.com" in emails
    assert "carol@test.com" not in emails
