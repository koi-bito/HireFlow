"""
tests/test_agents.py
---------------------
Mocked integration tests for all four HireFlow agents.

Replaces the stale ADK-era tests that imported deleted agent sub-packages
(agents.jd_analyst, agents.matcher, agents.outreach). These tests run
fully offline using unittest.mock — no API key, no network.
"""

import json
from unittest.mock import MagicMock, patch

from agents.candidate_matcher import run_candidate_matcher
from agents.interview_scheduler import run_interview_scheduler
from agents.jd_parser import run_jd_parser
from agents.outreach_composer import run_outreach_composer

# ── Shared fixtures ───────────────────────────────────────────────────────────

SAMPLE_JD = "Senior Python Developer with 5+ years experience in ML and FastAPI."

PARSED_JD_JSON = json.dumps({
    "job_title": "Senior Python Developer",
    "required_skills": ["Python", "ML", "FastAPI"],
    "preferred_skills": ["Docker"],
    "years_of_experience": "5+",
    "location": "Remote",
    "salary_range": None,
    "employment_type": "Full-time",
})

PARSED_JD = json.loads(PARSED_JD_JSON)

SCORE_JSON = json.dumps({
    "score": 85,
    "match_reasons": ["Strong Python", "ML background", "FastAPI experience"],
    "flag": "strong_match",
})

OUTREACH_JSON = json.dumps({
    "subject": "Great opportunity for you",
    "body": "Hi, we think you'd be a perfect fit. Let's connect!",
})


def _make_state(**overrides):
    state = {
        "raw_jd": SAMPLE_JD,
        "candidates_csv_path": "data/sample_candidates.csv",
        "parsed_jd": None,
        "ranked_candidates": None,
        "outreach_messages": None,
        "schedule": None,
        "errors": [],
    }
    state.update(overrides)
    return state


# ── Agent 1: JD Parser ────────────────────────────────────────────────────────

@patch("agents.jd_parser._client")
def test_jd_parser_extracts_role(mock_client):
    mock_resp = MagicMock()
    mock_resp.text = PARSED_JD_JSON
    mock_client.models.generate_content.return_value = mock_resp

    result = run_jd_parser(_make_state())
    assert result["parsed_jd"]["job_title"] == "Senior Python Developer"
    assert "Python" in result["parsed_jd"]["required_skills"]


@patch("agents.jd_parser._client")
def test_jd_parser_handles_malformed_response(mock_client):
    mock_resp = MagicMock()
    mock_resp.text = "This is not JSON at all"
    mock_client.models.generate_content.return_value = mock_resp

    result = run_jd_parser(_make_state())
    assert result["parsed_jd"]["required_skills"] == []  # safe default
    assert len(result["errors"]) > 0


# ── Agent 2: Candidate Matcher ────────────────────────────────────────────────

@patch("agents.candidate_matcher._client")
def test_matcher_returns_ranked_list(mock_client):
    mock_resp = MagicMock()
    mock_resp.text = SCORE_JSON
    mock_client.models.generate_content.return_value = mock_resp

    result = run_candidate_matcher(_make_state(parsed_jd=PARSED_JD))
    ranked = result["ranked_candidates"]
    assert len(ranked) > 0
    assert all("name" in c and "score" in c for c in ranked)


@patch("agents.candidate_matcher._client")
def test_matcher_scores_sorted_descending(mock_client):
    scores = iter([90, 60, 30, 80, 70, 50, 40, 20, 85, 10])

    def side_effect(**kwargs):
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({
            "score": next(scores, 50),
            "match_reasons": ["r1"], "flag": "partial_match",
        })
        return mock_resp

    mock_client.models.generate_content.side_effect = side_effect

    result = run_candidate_matcher(_make_state(parsed_jd=PARSED_JD))
    out = [c["score"] for c in result["ranked_candidates"]]
    assert out == sorted(out, reverse=True)


# ── Agent 3: Outreach Composer ────────────────────────────────────────────────

@patch("agents.outreach_composer._client")
def test_outreach_mentions_candidate(mock_client):
    mock_resp = MagicMock()
    mock_resp.text = OUTREACH_JSON
    mock_client.models.generate_content.return_value = mock_resp

    ranked = [{
        "name": "Alice Sharma", "email": "alice@example.com",
        "score": 92, "flag": "strong_match",
        "match_reasons": ["Python expert"],
    }]
    result = run_outreach_composer(_make_state(
        parsed_jd=PARSED_JD, ranked_candidates=ranked,
    ))
    assert len(result["outreach_messages"]) == 1
    assert result["outreach_messages"][0]["candidate_name"] == "Alice Sharma"


# ── Agent 4: Interview Scheduler ─────────────────────────────────────────────

@patch("agents.interview_scheduler._client")
def test_scheduler_generates_slots(mock_client):
    mock_resp = MagicMock()
    mock_resp.text = "Strong candidate — prioritise."
    mock_client.models.generate_content.return_value = mock_resp

    ranked = [{
        "name": "Alice Sharma", "email": "alice@example.com",
        "score": 92, "flag": "strong_match",
        "match_reasons": ["Python expert"],
    }]
    result = run_interview_scheduler(_make_state(ranked_candidates=ranked))
    assert len(result["schedule"]) == 1
    assert len(result["schedule"][0]["slots"]) == 3


def test_scheduler_skips_weak_candidates():
    ranked = [{
        "name": "Bob", "email": "bob@example.com",
        "score": 20, "flag": "weak_match", "match_reasons": [],
    }]
    result = run_interview_scheduler(_make_state(ranked_candidates=ranked))
    assert result["schedule"] == []
