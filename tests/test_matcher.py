"""
tests/test_matcher.py
----------------------
Unit and regression tests for Agent 2 - Candidate Matcher (v3).

Covers:
- Hybrid, embedding, and keyword retrieval modes
- Reciprocal Rank Fusion (RRF) and score blending
- Soft boosts: experience alignment, remote/location match
- Explainability: grounded matched and missing skills
- New additive fields: retrieval_score, llm_score, matched_skills, missing_skills
- Edge cases: empty skills, empty pool, single candidate, duplicates, long summaries, embedding failure
"""

import csv
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import config
from agents.candidate_matcher import (
    _keyword_overlap,
    build_jd_query_text,
    calculate_experience_boost,
    calculate_location_boost,
    extract_skill_matches,
    run_candidate_matcher,
)
from retrieval.embedder import FakeEmbedder

SAMPLE_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample_candidates.csv")

PARSED_JD = {
    "job_title": "Senior ML Engineer",
    "required_skills": ["Python", "machine learning", "FastAPI"],
    "preferred_skills": ["Docker", "Kubernetes"],
    "years_of_experience": "5+",
    "location": "Remote",
    "salary_range": "$150k",
    "employment_type": "Full-time",
}

VALID_SCORE_RESPONSE = json.dumps({
    "score": 85,
    "match_reasons": ["Demonstrated Python expertise", "FastAPI web services", "Production ML experience"],
    "flag": "strong_match",
})


def _make_state(csv_path=SAMPLE_CSV_PATH, parsed_jd=PARSED_JD, errors=None):
    return {
        "raw_jd": "dummy",
        "candidates_csv_path": csv_path,
        "parsed_jd": parsed_jd,
        "ranked_candidates": None,
        "outreach_messages": None,
        "schedule": None,
        "errors": errors or [],
    }


# ── Keyword Overlap & Skill Extraction Tests ─────────────────────────────────

def test_keyword_overlap_full_match():
    assert _keyword_overlap(["Python", "FastAPI"], "Python, FastAPI, Docker") == 1.0


def test_keyword_overlap_partial():
    ratio = _keyword_overlap(["Python", "FastAPI", "Go"], "Python, Django")
    assert 0 < ratio < 1.0


def test_keyword_overlap_empty_required_skills():
    assert _keyword_overlap([], "Python, FastAPI") == 1.0


def test_extract_skill_matches():
    req = ["Python", "PyTorch", "Kubernetes", "SQL"]
    cand_skills = "Python, PyTorch, Docker"
    summary = "Experienced in relational database design with SQL."

    matched, missing = extract_skill_matches(req, cand_skills, summary)
    assert "Python" in matched
    assert "PyTorch" in matched
    assert "SQL" in matched
    assert "Kubernetes" in missing


# ── Boost Calculation Tests ──────────────────────────────────────────────────

def test_experience_boost():
    # 5+ years required, candidate has 6
    boost = calculate_experience_boost("5+ years", 6)
    assert boost == config.EXPERIENCE_BOOST_WEIGHT

    # 5+ years required, candidate has 2 (under 70% threshold)
    boost_low = calculate_experience_boost("5+ years", 2)
    assert boost_low == 0.0


def test_location_boost():
    # Remote match
    assert calculate_location_boost("Remote", "Austin") == config.LOCATION_BOOST_WEIGHT
    assert calculate_location_boost("New York", "Remote") == config.LOCATION_BOOST_WEIGHT
    assert calculate_location_boost("San Francisco", "San Francisco") == config.LOCATION_BOOST_WEIGHT
    assert calculate_location_boost("San Francisco", "London") == 0.0


def test_build_jd_query_text():
    text = build_jd_query_text(PARSED_JD)
    assert "Senior ML Engineer" in text
    assert "FastAPI" in text
    assert "Remote" in text


# ── Matcher Modes & Output Shape ──────────────────────────────────────────────

@patch("agents.candidate_matcher._client")
def test_all_output_dicts_contain_required_and_additive_keys(mock_model):
    mock_resp = MagicMock()
    mock_resp.text = VALID_SCORE_RESPONSE
    mock_model.models.generate_content.return_value = mock_resp

    fake_embedder = FakeEmbedder(dimension=32)
    result = run_candidate_matcher(_make_state(), embedder=fake_embedder)
    ranked = result["ranked_candidates"]

    assert len(ranked) > 0
    for candidate in ranked:
        # Original required keys
        assert "name" in candidate
        assert "email" in candidate
        assert "score" in candidate
        assert "flag" in candidate
        assert "match_reasons" in candidate
        assert len(candidate["match_reasons"]) == 3

        # New additive keys
        assert "retrieval_score" in candidate
        assert "llm_score" in candidate
        assert "matched_skills" in candidate
        assert "missing_skills" in candidate
        assert 0 <= candidate["score"] <= 100
        assert 0.0 <= candidate["retrieval_score"] <= 1.0


@patch("agents.candidate_matcher._client")
def test_output_sorted_by_score_descending(mock_model):
    mock_resp = MagicMock()
    mock_resp.text = VALID_SCORE_RESPONSE
    mock_model.models.generate_content.return_value = mock_resp

    fake_embedder = FakeEmbedder(dimension=32)
    result = run_candidate_matcher(_make_state(), embedder=fake_embedder)
    scores = [c["score"] for c in result["ranked_candidates"]]
    assert scores == sorted(scores, reverse=True)


@patch("agents.candidate_matcher._client")
def test_keyword_matcher_mode(mock_model):
    mock_resp = MagicMock()
    mock_resp.text = VALID_SCORE_RESPONSE
    mock_model.models.generate_content.return_value = mock_resp

    with patch.object(config, "MATCHER_MODE", "keyword"):
        result = run_candidate_matcher(_make_state())
        ranked = result["ranked_candidates"]
        assert len(ranked) > 0
        # Alice has Python, ML, FastAPI so should rank at the top
        assert ranked[0]["name"] == "Alice Sharma"


@patch("agents.candidate_matcher._client")
def test_embedding_matcher_mode(mock_model):
    mock_resp = MagicMock()
    mock_resp.text = VALID_SCORE_RESPONSE
    mock_model.models.generate_content.return_value = mock_resp

    fake_embedder = FakeEmbedder(dimension=32)
    with patch.object(config, "MATCHER_MODE", "embedding"):
        result = run_candidate_matcher(_make_state(), embedder=fake_embedder)
        assert len(result["ranked_candidates"]) > 0


# ── Edge Cases ───────────────────────────────────────────────────────────────

def test_empty_candidate_pool():
    # Empty CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=["name", "email", "skills", "years_experience", "location", "summary"])
        writer.writeheader()
        tmp_name = tmp.name

    try:
        result = run_candidate_matcher(_make_state(csv_path=tmp_name))
        assert result["ranked_candidates"] == []
        assert any("empty" in e.lower() for e in result["errors"])
    finally:
        os.unlink(tmp_name)


@patch("agents.candidate_matcher._client")
def test_single_candidate(mock_model):
    mock_resp = MagicMock()
    mock_resp.text = VALID_SCORE_RESPONSE
    mock_model.models.generate_content.return_value = mock_resp

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=["name", "email", "skills", "years_experience", "location", "summary"])
        writer.writeheader()
        writer.writerow({
            "name": "Solo Dev",
            "email": "solo@example.com",
            "skills": "Python, FastAPI",
            "years_experience": "3",
            "location": "Remote",
            "summary": "Building web apps.",
        })
        tmp_name = tmp.name

    try:
        fake_embedder = FakeEmbedder(dimension=16)
        result = run_candidate_matcher(_make_state(csv_path=tmp_name), embedder=fake_embedder)
        assert len(result["ranked_candidates"]) == 1
        assert result["ranked_candidates"][0]["name"] == "Solo Dev"
    finally:
        os.unlink(tmp_name)


@patch("agents.candidate_matcher._client")
def test_duplicate_emails_are_deduplicated(mock_model):
    mock_resp = MagicMock()
    mock_resp.text = VALID_SCORE_RESPONSE
    mock_model.models.generate_content.return_value = mock_resp

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=["name", "email", "skills", "years_experience", "location", "summary"])
        writer.writeheader()
        writer.writerow({"name": "First Copy", "email": "dup@example.com", "skills": "Python", "years_experience": "3", "location": "Remote", "summary": "S1"})
        writer.writerow({"name": "Second Copy", "email": "dup@example.com", "skills": "Java", "years_experience": "5", "location": "NYC", "summary": "S2"})
        tmp_name = tmp.name

    try:
        fake_embedder = FakeEmbedder(dimension=16)
        result = run_candidate_matcher(_make_state(csv_path=tmp_name), embedder=fake_embedder)
        assert len(result["ranked_candidates"]) == 1
        assert result["ranked_candidates"][0]["name"] == "First Copy"
    finally:
        os.unlink(tmp_name)


@patch("agents.candidate_matcher._client")
def test_blank_skills_and_summary_do_not_crash(mock_model):
    mock_resp = MagicMock()
    mock_resp.text = VALID_SCORE_RESPONSE
    mock_model.models.generate_content.return_value = mock_resp

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=["name", "email", "skills", "years_experience", "location", "summary"])
        writer.writeheader()
        writer.writerow({"name": "Blank User", "email": "blank@example.com", "skills": "", "years_experience": "", "location": "", "summary": ""})
        tmp_name = tmp.name

    try:
        fake_embedder = FakeEmbedder(dimension=16)
        result = run_candidate_matcher(_make_state(csv_path=tmp_name), embedder=fake_embedder)
        assert len(result["ranked_candidates"]) == 1
        assert result["ranked_candidates"][0]["score"] >= 0
    finally:
        os.unlink(tmp_name)


@patch("agents.candidate_matcher._client")
def test_long_summary_truncated_and_non_english(mock_model):
    mock_resp = MagicMock()
    mock_resp.text = VALID_SCORE_RESPONSE
    mock_model.models.generate_content.return_value = mock_resp

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=["name", "email", "skills", "years_experience", "location", "summary"])
        writer.writeheader()
        long_summary = "Développeur Python spécialisé dans l'apprentissage automatique et le NLP. " * 50
        writer.writerow({"name": "Jean Dupont", "email": "jean@example.fr", "skills": "Python, NLP", "years_experience": "5", "location": "Paris", "summary": long_summary})
        tmp_name = tmp.name

    try:
        fake_embedder = FakeEmbedder(dimension=16)
        result = run_candidate_matcher(_make_state(csv_path=tmp_name), embedder=fake_embedder)
        assert len(result["ranked_candidates"]) == 1
        assert result["ranked_candidates"][0]["name"] == "Jean Dupont"
    finally:
        os.unlink(tmp_name)


@patch("agents.candidate_matcher._client")
def test_embedding_service_down_falls_back_to_keyword(mock_model):
    mock_resp = MagicMock()
    mock_resp.text = VALID_SCORE_RESPONSE
    mock_model.models.generate_content.return_value = mock_resp

    broken_embedder = MagicMock()
    broken_embedder.embed_texts.side_effect = RuntimeError("Embedding service unavailable 503")

    result = run_candidate_matcher(_make_state(), embedder=broken_embedder)
    # Pipeline still produces ranked candidates via keyword fallback
    assert len(result["ranked_candidates"]) > 0
    # Error was recorded in state["errors"]
    assert any("Embedding service failed" in err for err in result["errors"])


def test_missing_csv_columns_returns_error():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as tmp:
        writer = csv.DictWriter(tmp, fieldnames=["name", "bad_col"])
        writer.writeheader()
        writer.writerow({"name": "Test", "bad_col": "value"})
        tmp_name = tmp.name

    try:
        state = _make_state(csv_path=tmp_name)
        result = run_candidate_matcher(state)
        assert result["ranked_candidates"] == []
        assert len(result["errors"]) > 0
    finally:
        os.unlink(tmp_name)


def test_nonexistent_csv_falls_back_to_sample():
    with patch("agents.candidate_matcher._client") as mock_model:
        mock_response = MagicMock()
        mock_response.text = VALID_SCORE_RESPONSE
        mock_model.models.generate_content.return_value = mock_response

        state = _make_state(csv_path="nonexistent/path/to/file.csv")
        result = run_candidate_matcher(state)
        assert "ranked_candidates" in result
