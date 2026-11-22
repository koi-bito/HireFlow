"""
tests/test_api.py
------------------
Direct tests for FastAPI route handlers and request/response models:
- health_check()
- run_pipeline()
- get_results()
- submit_feedback()
- get_runs()
- recruiter_chat()
"""

import json
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from app import (
    ChatRequest,
    FeedbackRequest,
    RunRequest,
    get_results,
    get_runs,
    health_check,
    recruiter_chat,
    run_pipeline,
    submit_feedback,
)


def test_health_check_endpoint():
    res = health_check()
    assert res.status in {"ok", "degraded"}
    assert res.version == "3.0.0"
    assert res.database == "ok"


def test_post_feedback_endpoint():
    req = FeedbackRequest(
        candidate_email="alice@example.com",
        job_title="Senior ML Engineer",
        action="shortlisted",
    )
    res = submit_feedback(req)
    assert res["status"] == "recorded"
    assert res["feedback"]["action"] == "shortlisted"


def test_post_feedback_invalid_action():
    with pytest.raises((HTTPException, ValueError)):
        # Pydantic or handler validation
        req = FeedbackRequest(
            candidate_email="bob@example.com",
            job_title="ML Engineer",
            action="promoted",  # invalid action
        )
        submit_feedback(req)


def test_get_runs_endpoint():
    res = get_runs(limit=5, offset=0)
    assert "runs" in res
    assert "total" in res


def test_post_chat_endpoint():
    req = ChatRequest(
        message="Find candidates for Senior Python Engineer",
        session_id="api-session-123",
    )
    res = recruiter_chat(req)
    assert res["session_id"] == "api-session-123"
    assert "reply" in res
    assert "tool_calls" in res


@patch("agents.jd_parser._client")
@patch("agents.candidate_matcher._client")
@patch("agents.outreach_composer._client")
@patch("agents.interview_scheduler._client")
def test_run_and_get_results_round_trip(mock_sched, mock_out, mock_match, mock_jd):
    mock_jd.models.generate_content.return_value.text = json.dumps({
        "job_title": "ML Engineer", "required_skills": ["Python"],
        "preferred_skills": [], "years_of_experience": "3+", "location": "Remote",
        "salary_range": None, "employment_type": "Full-time",
    })
    mock_match.models.generate_content.return_value.text = json.dumps({
        "score": 90, "match_reasons": ["Python expert"], "flag": "strong_match",
    })
    mock_out.models.generate_content.return_value.text = json.dumps({
        "subject": "Role", "body": "Hello",
    })
    mock_sched.models.generate_content.return_value.text = "Good candidate."

    req = RunRequest(job_description="Senior ML Engineer with Python.")
    run_data = run_pipeline(req)
    run_id = run_data["run_id"]
    assert len(run_data["ranked_candidates"]) > 0

    # Retrieve from DB / memory
    res_data = get_results(run_id)
    assert res_data["run_id"] == run_id
    assert res_data["ranked_candidates"][0]["name"] == run_data["ranked_candidates"][0]["name"]
