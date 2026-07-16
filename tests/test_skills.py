import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.tfidf_matcher import match_candidates_to_jd


def test_basic_ranking():
    jd = "Python machine learning NLP developer."
    candidates = [
        {"name": "Alice", "skills": ["Python", "machine learning", "NLP"]},
        {"name": "Bob", "skills": ["Java", "Spring Boot"]}
    ]
    result = match_candidates_to_jd(jd, json.dumps(candidates))
    assert result["ranked_candidates"][0]["name"] == "Alice"


def test_empty_candidates():
    result = match_candidates_to_jd("Python developer", json.dumps([]))
    assert "error" in result


def test_invalid_json():
    result = match_candidates_to_jd("Python developer", "not valid json")
    assert "error" in result


def test_all_candidates_returned():
    jd = "Python developer"
    candidates = [
        {"name": "A", "skills": ["Python"]},
        {"name": "B", "skills": ["Java"]},
        {"name": "C", "skills": ["Ruby"]}
    ]
    result = match_candidates_to_jd(jd, json.dumps(candidates))
    assert result["total"] == 3


def test_scores_between_zero_and_one():
    jd = "Python developer"
    candidates = [{"name": "Alice", "skills": ["Python", "Django"]}]
    result = match_candidates_to_jd(jd, json.dumps(candidates))
    for c in result["ranked_candidates"]:
        assert 0.0 <= c["match_score"] <= 1.0
