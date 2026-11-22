"""
tests/test_scheduler.py
------------------------
Unit tests for Agent 4 - Interview Scheduler.
"""

from unittest.mock import MagicMock, patch

from agents.interview_scheduler import _next_n_business_days, run_interview_scheduler

# ── Fixtures ───────────────────────────────────────────────────────────────────

STRONG_CANDIDATES = [
    {
        "name": "Alice Sharma",
        "email": "alice@example.com",
        "score": 92,
        "flag": "strong_match",
        "match_reasons": ["Expert NLP", "FastAPI experience"],
    },
    {
        "name": "Sarah O'Brien",
        "email": "sarah@example.com",
        "score": 78,
        "flag": "partial_match",
        "match_reasons": ["Feature engineering"],
    },
]

WEAK_ONLY = [
    {
        "name": "Bob Zhang",
        "email": "bob@example.com",
        "score": 20,
        "flag": "weak_match",
        "match_reasons": [],
    }
]


def _make_state(candidates=STRONG_CANDIDATES):
    return {
        "raw_jd": "dummy",
        "candidates_csv_path": "data/sample_candidates.csv",
        "parsed_jd": {"job_title": "ML Engineer"},
        "ranked_candidates": candidates,
        "outreach_messages": [],
        "schedule": None,
        "errors": [],
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_business_days_excludes_weekends():
    """_next_n_business_days must never return Saturday (5) or Sunday (6)."""
    days = _next_n_business_days(n=10)
    for d in days:
        assert d.weekday() < 5, f"{d} is a weekend day"


def test_business_days_count():
    """Must return exactly n business days."""
    for n in [1, 3, 5, 7]:
        assert len(_next_n_business_days(n=n)) == n


@patch("agents.interview_scheduler._client")
def test_schedule_contains_required_keys(mock_model):
    """Each schedule entry must have candidate_name, email, flag, slots, notes."""
    mock_response = MagicMock()
    mock_response.text = "Strong candidate - prioritise scheduling."
    mock_model.models.generate_content.return_value = mock_response

    result = run_interview_scheduler(_make_state())
    for entry in result["schedule"]:
        assert "candidate_name" in entry
        assert "email" in entry
        assert "flag" in entry
        assert "slots" in entry
        assert "notes" in entry


@patch("agents.interview_scheduler._client")
def test_each_candidate_has_three_slots(mock_model):
    """Each candidate should have exactly 3 interview slot options."""
    mock_response = MagicMock()
    mock_response.text = "Good fit."
    mock_model.models.generate_content.return_value = mock_response

    result = run_interview_scheduler(_make_state())
    for entry in result["schedule"]:
        assert len(entry["slots"]) == 3, f"{entry['candidate_name']} does not have 3 slots"


@patch("agents.interview_scheduler._client")
def test_slots_have_correct_structure(mock_model):
    """Each slot must have date, time, and duration_min fields."""
    mock_response = MagicMock()
    mock_response.text = "Schedule soon."
    mock_model.models.generate_content.return_value = mock_response

    result = run_interview_scheduler(_make_state())
    for entry in result["schedule"]:
        for slot in entry["slots"]:
            assert "date" in slot
            assert "time" in slot
            assert "duration_min" in slot
            assert slot["duration_min"] == 45


def test_weak_match_candidates_not_scheduled():
    """Candidates flagged 'weak_match' must not appear in the schedule."""
    result = run_interview_scheduler(_make_state(candidates=WEAK_ONLY))
    assert result["schedule"] == []
    assert len(result["errors"]) > 0


@patch("agents.interview_scheduler._client")
def test_llm_note_failure_uses_fallback(mock_model):
    """If the LLM fails to generate a note, a safe fallback string is used."""
    mock_model.models.generate_content.side_effect = Exception("API error")

    result = run_interview_scheduler(_make_state(candidates=[STRONG_CANDIDATES[0]]))
    assert len(result["schedule"]) == 1
    assert result["schedule"][0]["notes"] != ""
