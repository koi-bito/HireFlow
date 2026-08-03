"""
tests/test_pipeline.py
-----------------------
Integration test — runs the full 4-agent LangGraph pipeline with mocked
Gemini API calls and the real sample CSV.

This test validates end-to-end state propagation without hitting the API.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from pipeline.graph import build_pipeline

# ── Sample data ────────────────────────────────────────────────────────────────

SAMPLE_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_candidates.csv")

REALISTIC_JD = """
We are looking for a Senior Machine Learning Engineer to join our AI team.
Requirements: 5+ years of Python, experience with TensorFlow or PyTorch, FastAPI, Docker.
Location: Remote. Salary: $150,000–$180,000 USD. Full-time.
"""

PARSED_JD_RESPONSE = json.dumps({
    "job_title": "Senior ML Engineer",
    "required_skills": ["Python", "machine learning", "FastAPI", "Docker"],
    "preferred_skills": ["Kubernetes"],
    "years_of_experience": "5+ years",
    "location": "Remote",
    "salary_range": "$150,000–$180,000 USD",
    "employment_type": "Full-time",
})

SCORING_RESPONSE = json.dumps({
    "score": 88,
    "match_reasons": ["Strong Python skills", "ML experience", "FastAPI use"],
    "flag": "strong_match",
})

OUTREACH_RESPONSE = json.dumps({
    "subject": "Exciting ML opportunity for you",
    "body": "Hi there! We think you'd be a great fit for our Senior ML role. Let's connect!",
})


def _make_initial_state():
    return {
        "raw_jd": REALISTIC_JD,
        "candidates_csv_path": SAMPLE_CSV_PATH,
        "parsed_jd": None,
        "ranked_candidates": None,
        "outreach_messages": None,
        "schedule": None,
        "errors": [],
    }


# ── Integration test ───────────────────────────────────────────────────────────

def _mock_generate(*args, **kwargs) -> MagicMock:
    """Return appropriate mock based on the prompt content (contents kwarg)."""
    prompt = kwargs.get("contents", args[0] if args else "")
    mock_resp = MagicMock()
    if "recruitment data extractor" in prompt:
        mock_resp.text = PARSED_JD_RESPONSE
    elif "senior technical recruiter" in prompt:
        mock_resp.text = SCORING_RESPONSE
    elif "recruiter outreach email" in prompt:
        mock_resp.text = OUTREACH_RESPONSE
    else:
        mock_resp.text = "Strong candidate — schedule soon."
    return mock_resp


@patch("agents.jd_parser._client")
@patch("agents.candidate_matcher._client")
@patch("agents.outreach_composer._client")
@patch("agents.interview_scheduler._client")
def test_full_pipeline_runs_end_to_end(mock_sched, mock_out, mock_match, mock_jd):
    """The pipeline must execute all 4 agents and return populated output."""
    mock_jd.models.generate_content.side_effect = _mock_generate
    mock_match.models.generate_content.side_effect = _mock_generate
    mock_out.models.generate_content.side_effect = _mock_generate
    mock_sched.models.generate_content.side_effect = _mock_generate

    pipeline = build_pipeline()
    final_state = pipeline.invoke(_make_initial_state())

    # Agent 1 output
    assert final_state["parsed_jd"] is not None
    assert final_state["parsed_jd"]["job_title"] == "Senior ML Engineer"

    # Agent 2 output
    assert isinstance(final_state["ranked_candidates"], list)
    assert len(final_state["ranked_candidates"]) > 0

    # Agent 3 output
    assert isinstance(final_state["outreach_messages"], list)
    assert len(final_state["outreach_messages"]) > 0

    # Agent 4 output
    assert isinstance(final_state["schedule"], list)
    assert len(final_state["schedule"]) > 0


@patch("agents.jd_parser._client")
@patch("agents.candidate_matcher._client")
@patch("agents.outreach_composer._client")
@patch("agents.interview_scheduler._client")
def test_pipeline_state_fields_are_populated(mock_sched, mock_out, mock_match, mock_jd):
    """All state fields should be populated (not None) after a successful run."""
    mock_jd.models.generate_content.side_effect = _mock_generate
    mock_match.models.generate_content.side_effect = _mock_generate
    mock_out.models.generate_content.side_effect = _mock_generate
    mock_sched.models.generate_content.side_effect = _mock_generate

    pipeline = build_pipeline()
    final_state = pipeline.invoke(_make_initial_state())

    assert final_state.get("parsed_jd") is not None
    assert final_state.get("ranked_candidates") is not None
    assert final_state.get("outreach_messages") is not None
    assert final_state.get("schedule") is not None


@patch("agents.jd_parser._client")
@patch("agents.candidate_matcher._client")
@patch("agents.outreach_composer._client")
@patch("agents.interview_scheduler._client")
def test_ranked_candidates_sorted_in_final_state(mock_sched, mock_out, mock_match, mock_jd):
    """Final ranked_candidates must be sorted by score descending."""
    call_idx = 0
    scores = [70, 90, 40, 80, 60, 55, 45, 85, 75, 30]

    def side_effect(*args, **kwargs):
        nonlocal call_idx
        prompt = kwargs.get("contents", args[0] if args else "")
        mock_resp = MagicMock()
        if "senior technical recruiter" in prompt:
            mock_resp.text = json.dumps({
                "score": scores[call_idx % len(scores)],
                "match_reasons": ["r1", "r2", "r3"],
                "flag": "strong_match",
            })
            call_idx += 1
        else:
            mock_resp.text = _mock_generate(prompt).text
        return mock_resp

    mock_jd.models.generate_content.side_effect = _mock_generate
    mock_match.models.generate_content.side_effect = side_effect
    mock_out.models.generate_content.side_effect = _mock_generate
    mock_sched.models.generate_content.side_effect = _mock_generate

    pipeline = build_pipeline()
    final_state = pipeline.invoke(_make_initial_state())

    scores_out = [c["score"] for c in final_state["ranked_candidates"]]
    assert scores_out == sorted(scores_out, reverse=True)
