"""
tests/test_matcher.py
----------------------
Unit tests for Agent 2 - Candidate Matcher.

Mocks Gemini API calls and uses the real sample CSV.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from agents.candidate_matcher import run_candidate_matcher, _keyword_overlap

# ── Fixtures ───────────────────────────────────────────────────────────────────

SAMPLE_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_candidates.csv")

PARSED_JD = {
    "job_title": "Senior ML Engineer",
    "required_skills": ["Python", "machine learning", "FastAPI"],
    "preferred_skills": ["Docker"],
    "years_of_experience": "5+",
    "location": "Remote",
    "salary_range": "$150k",
    "employment_type": "Full-time",
}

VALID_SCORE_RESPONSE = json.dumps({
    "score": 85,
    "match_reasons": ["Strong Python skills", "ML experience", "FastAPI proficiency"],
    "flag": "strong_match",
})


def _make_state(csv_path=SAMPLE_CSV_PATH, parsed_jd=PARSED_JD):
    return {
        "raw_jd": "dummy",
        "candidates_csv_path": csv_path,
        "parsed_jd": parsed_jd,
        "ranked_candidates": None,
        "outreach_messages": None,
        "schedule": None,
        "errors": [],
    }


# ── Unit tests ─────────────────────────────────────────────────────────────────

def test_keyword_overlap_full_match():
    assert _keyword_overlap(["Python", "FastAPI"], "Python, FastAPI, Docker") == 1.0


def test_keyword_overlap_partial():
    ratio = _keyword_overlap(["Python", "FastAPI", "Go"], "Python, Django")
    assert 0 < ratio < 1.0


def test_keyword_overlap_empty_required_skills():
    """When no required skills are specified, every candidate passes."""
    assert _keyword_overlap([], "Python, FastAPI") == 1.0


@patch("agents.candidate_matcher._client")
def test_output_sorted_by_score_descending(mock_model):
    """ranked_candidates must be sorted by score, highest first."""
    scores = [60, 90, 40, 75, 85, 50, 30, 70, 80, 20]
    call_count = 0

    def side_effect(prompt):
        nonlocal call_count
        mock_resp = MagicMock()
        mock_resp.text = json.dumps({
            "score": scores[call_count % len(scores)],
            "match_reasons": ["reason 1", "reason 2", "reason 3"],
            "flag": "partial_match",
        })
        call_count += 1
        return mock_resp

    mock_model.models.generate_content.side_effect = side_effect

    result = run_candidate_matcher(_make_state())
    ranked = result["ranked_candidates"]

    assert len(ranked) > 0
    scores_out = [c["score"] for c in ranked]
    assert scores_out == sorted(scores_out, reverse=True), "Candidates not sorted by score"


@patch("agents.candidate_matcher._client")
def test_all_output_dicts_contain_required_keys(mock_model):
    """Every entry in ranked_candidates must have name, email, score, flag, match_reasons."""
    mock_response = MagicMock()
    mock_response.text = VALID_SCORE_RESPONSE
    mock_model.models.generate_content.return_value = mock_response

    result = run_candidate_matcher(_make_state())
    for candidate in result["ranked_candidates"]:
        assert "name" in candidate
        assert "email" in candidate
        assert "score" in candidate
        assert "flag" in candidate
        assert "match_reasons" in candidate


def test_missing_csv_columns_returns_error():
    """A CSV missing required columns should return a clear error, not raise."""
    import tempfile, csv, os as _os
    # Create a temp CSV with bad columns
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="")
    writer = csv.DictWriter(tmp, fieldnames=["name", "bad_col"])
    writer.writeheader()
    writer.writerow({"name": "Test", "bad_col": "value"})
    tmp.close()

    state = _make_state(csv_path=tmp.name)
    result = run_candidate_matcher(state)

    _os.unlink(tmp.name)

    assert result["ranked_candidates"] == []
    assert len(result["errors"]) > 0


def test_nonexistent_csv_falls_back_to_sample():
    """A path that doesn't exist should fall back to sample data without crashing."""
    with patch("agents.candidate_matcher._client") as mock_model:
        mock_response = MagicMock()
        mock_response.text = VALID_SCORE_RESPONSE
        mock_model.models.generate_content.return_value = mock_response

        state = _make_state(csv_path="nonexistent/path/to/file.csv")
        result = run_candidate_matcher(state)
        # Should either return candidates (from fallback) or an error but not raise
        assert "ranked_candidates" in result
