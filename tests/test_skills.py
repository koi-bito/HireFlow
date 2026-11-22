"""
tests/test_skills.py
---------------------
Tests for the keyword-overlap skill used by the Candidate Matcher.

Replaces the original TF-IDF matcher tests that referenced a deleted
`tools/tfidf_matcher` module from the legacy Flask prototype.
"""

import pytest

from agents.candidate_matcher import _keyword_overlap


def test_full_overlap_returns_one():
    """All required skills present in candidate → 1.0."""
    assert _keyword_overlap(["Python", "FastAPI"], "Python, FastAPI, Docker") == 1.0


def test_partial_overlap():
    ratio = _keyword_overlap(["Python", "FastAPI", "Go"], "Python, Django")
    assert ratio == pytest.approx(1 / 3)


def test_no_overlap_returns_zero():
    assert _keyword_overlap(["Rust", "C++"], "Python, Django") == 0.0


def test_empty_required_skills_returns_one():
    """No required skills → every candidate passes."""
    assert _keyword_overlap([], "Python, FastAPI") == 1.0


def test_case_insensitive():
    assert _keyword_overlap(["python"], "Python, Django") == 1.0


