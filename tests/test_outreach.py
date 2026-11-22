"""
tests/test_outreach.py
-----------------------
Unit tests for Agent 3 - Outreach Composer.
"""

import json
from unittest.mock import MagicMock, patch

from agents.outreach_composer import run_outreach_composer

# ── Fixtures ───────────────────────────────────────────────────────────────────

RANKED_CANDIDATES = [
    {
        "name": "Alice Sharma",
        "email": "alice@example.com",
        "score": 92,
        "flag": "strong_match",
        "match_reasons": ["Expert NLP engineer", "FastAPI experience", "5 years ML"],
    },
    {
        "name": "Sarah O'Brien",
        "email": "sarah@example.com",
        "score": 78,
        "flag": "strong_match",
        "match_reasons": ["Feature engineering", "A/B testing expertise", "Python skills"],
    },
    {
        "name": "Arjun Patel",
        "email": "arjun@example.com",
        "score": 55,
        "flag": "partial_match",
        "match_reasons": ["Python backend", "Docker experience"],
    },
]

VALID_EMAIL_RESPONSE = json.dumps({
    "subject": "Exciting ML Engineer opportunity - let's connect",
    "body": "Hi Alice, we came across your profile and were impressed by your NLP work. We'd love to chat about a Senior ML role that matches your background perfectly. Looking forward to connecting!",
})

PARSED_JD = {
    "job_title": "Senior ML Engineer",
    "required_skills": ["Python", "machine learning"],
}


def _make_state(ranked=RANKED_CANDIDATES, parsed_jd=PARSED_JD, errors=None):
    return {
        "raw_jd": "dummy",
        "candidates_csv_path": "data/sample_candidates.csv",
        "parsed_jd": parsed_jd,
        "ranked_candidates": ranked,
        "outreach_messages": None,
        "schedule": None,
        "errors": errors or [],
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

@patch("agents.outreach_composer._client")
def test_outreach_contains_required_keys(mock_model):
    """Each outreach message must have candidate_name, email, subject, body."""
    mock_response = MagicMock()
    mock_response.text = VALID_EMAIL_RESPONSE
    mock_model.models.generate_content.return_value = mock_response

    result = run_outreach_composer(_make_state())
    for msg in result["outreach_messages"]:
        assert "candidate_name" in msg
        assert "email" in msg
        assert "subject" in msg
        assert "body" in msg


@patch("agents.outreach_composer._client")
def test_outreach_limited_to_max_candidates(mock_model):
    """Should not compose more than OUTREACH_MAX_CANDIDATES emails (default 5)."""
    mock_response = MagicMock()
    mock_response.text = VALID_EMAIL_RESPONSE
    mock_model.models.generate_content.return_value = mock_response

    # Build 8 candidates
    many_candidates = [
        {"name": f"Candidate {i}", "email": f"c{i}@test.com",
         "score": 90 - i, "flag": "strong_match", "match_reasons": ["skill A"]}
        for i in range(8)
    ]
    result = run_outreach_composer(_make_state(ranked=many_candidates))
    assert len(result["outreach_messages"]) <= 5


def test_empty_candidates_list_returns_gracefully():
    """When ranked_candidates is empty, outreach_messages should be [] without raising."""
    result = run_outreach_composer(_make_state(ranked=[]))
    assert result["outreach_messages"] == []
    assert len(result["errors"]) > 0


@patch("agents.outreach_composer._client")
def test_llm_failure_falls_back_to_generic_email(mock_model):
    """If the LLM call fails, a safe fallback email should be generated."""
    mock_model.models.generate_content.side_effect = Exception("API quota exceeded")

    result = run_outreach_composer(_make_state(ranked=[RANKED_CANDIDATES[0]]))
    assert len(result["outreach_messages"]) == 1
    assert result["outreach_messages"][0]["subject"] != ""
    assert result["outreach_messages"][0]["body"] != ""


@patch("agents.outreach_composer._client")
def test_email_addresses_match_candidates(mock_model):
    """The email field in each outreach message must match the candidate's email."""
    mock_response = MagicMock()
    mock_response.text = VALID_EMAIL_RESPONSE
    mock_model.models.generate_content.return_value = mock_response

    result = run_outreach_composer(_make_state())
    name_to_email = {c["name"]: c["email"] for c in RANKED_CANDIDATES}
    for msg in result["outreach_messages"]:
        assert msg["email"] == name_to_email.get(msg["candidate_name"], msg["email"])
