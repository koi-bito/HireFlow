"""
tests/test_fusion.py
---------------------
Unit tests for Reciprocal Rank Fusion (RRF) and normalization.
"""

import pytest

from retrieval.fusion import min_max_normalize, reciprocal_rank_fusion


def test_rrf_empty():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[]]) == {}


def test_rrf_single_list():
    ranking = ["candidate_A", "candidate_B", "candidate_C"]
    scores = reciprocal_rank_fusion([ranking], k=60)

    # Rank 1 gets highest score, monotonically decreasing
    assert scores["candidate_A"] == 1.0  # normalized max
    assert scores["candidate_A"] > scores["candidate_B"] > scores["candidate_C"]


def test_rrf_multiple_lists_agreement():
    # Both lists rank A first, B second
    list1 = ["A", "B", "C"]
    list2 = ["A", "B", "D"]
    scores = reciprocal_rank_fusion([list1, list2], k=60)

    assert scores["A"] == 1.0
    assert scores["A"] > scores["B"]
    # Both C and D were rank 3 in one list and absent in the other
    assert pytest.approx(scores["C"]) == scores["D"]
    assert scores["B"] > scores["C"]


def test_rrf_disagreement_compromise():
    # List 1: A > B
    # List 2: B > A
    list1 = ["A", "B"]
    list2 = ["B", "A"]
    scores = reciprocal_rank_fusion([list1, list2], k=60)

    # Scores should be identical since symmetric
    assert pytest.approx(scores["A"]) == scores["B"]


def test_min_max_normalize():
    assert min_max_normalize({}) == {}
    assert min_max_normalize({"a": 10.0, "b": 20.0}) == {"a": 0.0, "b": 1.0}
    assert min_max_normalize({"a": 5.0, "b": 5.0}) == {"a": 1.0, "b": 1.0}
