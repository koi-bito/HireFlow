"""
tests/test_jd_parser.py
------------------------
Unit tests for Agent 1 - JD Parser.

These tests use mocking to avoid real API calls, ensuring CI can run
without a GOOGLE_API_KEY.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from agents.jd_parser import run_jd_parser, _SAFE_DEFAULT


# ── Fixtures ───────────────────────────────────────────────────────────────────

REALISTIC_JD = """
We are looking for a Senior Machine Learning Engineer to join our AI team.
The ideal candidate has 5+ years of experience with Python, TensorFlow or PyTorch,
and has built production ML systems. Experience with MLflow, FastAPI, and Docker is required.
Nice to have: Kubernetes, Spark. Location: Remote. Salary: $150,000–$180,000 USD. Full-time.
"""

VALID_LLM_RESPONSE = json.dumps({
    "job_title": "Senior Machine Learning Engineer",
    "required_skills": ["Python", "TensorFlow", "PyTorch", "MLflow", "FastAPI", "Docker"],
    "preferred_skills": ["Kubernetes", "Spark"],
    "years_of_experience": "5+ years",
    "location": "Remote",
    "salary_range": "$150,000–$180,000 USD",
    "employment_type": "Full-time",
})


def _make_state(raw_jd: str = REALISTIC_JD, errors: list | None = None) -> dict:
    return {
        "raw_jd": raw_jd,
        "candidates_csv_path": "data/sample_candidates.csv",
        "parsed_jd": None,
        "ranked_candidates": None,
        "outreach_messages": None,
        "schedule": None,
        "errors": errors or [],
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

@patch("agents.jd_parser._client")
def test_all_keys_present_in_output(mock_model):
    """Given a realistic JD, all expected keys must be present in parsed_jd."""
    mock_response = MagicMock()
    mock_response.text = VALID_LLM_RESPONSE
    mock_model.models.generate_content.return_value = mock_response

    result = run_jd_parser(_make_state())
    parsed = result["parsed_jd"]

    expected_keys = {
        "job_title", "required_skills", "preferred_skills",
        "years_of_experience", "location", "salary_range", "employment_type"
    }
    assert expected_keys.issubset(parsed.keys()), f"Missing keys: {expected_keys - parsed.keys()}"


@patch("agents.jd_parser._client")
def test_required_skills_is_always_a_list(mock_model):
    """parsed_jd['required_skills'] must be a list, never None."""
    mock_response = MagicMock()
    mock_response.text = VALID_LLM_RESPONSE
    mock_model.models.generate_content.return_value = mock_response

    result = run_jd_parser(_make_state())
    assert isinstance(result["parsed_jd"]["required_skills"], list)


def test_empty_jd_does_not_raise():
    """Given an empty JD, the function must return safe defaults without raising."""
    result = run_jd_parser(_make_state(raw_jd=""))

    assert result["parsed_jd"] is not None
    assert isinstance(result["parsed_jd"]["required_skills"], list)
    assert len(result["errors"]) > 0  # error should be logged


@patch("agents.jd_parser._client")
def test_malformed_llm_response_falls_back_to_safe_default(mock_model):
    """If the LLM returns non-JSON garbage, the safe default must be used."""
    mock_response = MagicMock()
    mock_response.text = "Sorry, I cannot help with that."
    mock_model.models.generate_content.return_value = mock_response

    result = run_jd_parser(_make_state())

    assert result["parsed_jd"]["required_skills"] == []
    assert len(result["errors"]) > 0


@patch("agents.jd_parser._client")
def test_errors_list_preserved_across_calls(mock_model):
    """Pre-existing errors in state must be carried forward, not overwritten."""
    mock_response = MagicMock()
    mock_response.text = VALID_LLM_RESPONSE
    mock_model.models.generate_content.return_value = mock_response

    pre_existing_error = "Some upstream error"
    result = run_jd_parser(_make_state(errors=[pre_existing_error]))

    assert pre_existing_error in result["errors"]
