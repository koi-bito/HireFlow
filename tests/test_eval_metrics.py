"""
tests/test_eval_metrics.py
---------------------------
Unit tests for information retrieval and ranking metrics verified against hand-computed examples.
"""

import math

import pytest

from evals.metrics import (
    dcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_precision_at_k_hand_computed():
    # Retrieved 5 items: ['c1', 'c2', 'c3', 'c4', 'c5']
    # Relevant: {'c1', 'c3', 'c9'}
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = {"c1", "c3", "c9"}

    # At k=1: ['c1'] -> 1 relevant / 1 = 1.0
    assert precision_at_k(retrieved, relevant, k=1) == 1.0

    # At k=2: ['c1', 'c2'] -> 1 relevant / 2 = 0.5
    assert precision_at_k(retrieved, relevant, k=2) == 0.5

    # At k=3: ['c1', 'c2', 'c3'] -> 2 relevant / 3 = 2/3
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)

    # At k=5: ['c1', 'c2', 'c3', 'c4', 'c5'] -> 2 relevant / 5 = 0.4
    assert precision_at_k(retrieved, relevant, k=5) == 0.4

    # k=0 or empty
    assert precision_at_k([], relevant, k=5) == 0.0
    assert precision_at_k(retrieved, relevant, k=0) == 0.0


def test_recall_at_k_hand_computed():
    retrieved = ["c1", "c2", "c3", "c4", "c5"]
    relevant = {"c1", "c3", "c9"}  # 3 total relevant items

    # At k=1: ['c1'] -> 1 / 3
    assert recall_at_k(retrieved, relevant, k=1) == pytest.approx(1 / 3)

    # At k=3: ['c1', 'c2', 'c3'] -> 2 / 3
    assert recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)

    # At k=5: ['c1', 'c2', 'c3', 'c4', 'c5'] -> 2 / 3 (since 'c9' was not retrieved)
    assert recall_at_k(retrieved, relevant, k=5) == pytest.approx(2 / 3)


def test_reciprocal_rank_and_mrr():
    # First relevant at rank 1 -> 1.0
    assert reciprocal_rank(["c1", "c2"], {"c1"}) == 1.0

    # First relevant at rank 2 -> 0.5
    assert reciprocal_rank(["c2", "c1"], {"c1"}) == 0.5

    # First relevant at rank 4 -> 0.25
    assert reciprocal_rank(["c2", "c3", "c4", "c1"], {"c1"}) == 0.25

    # No relevant item -> 0.0
    assert reciprocal_rank(["c2", "c3"], {"c1"}) == 0.0

    # MRR across 2 queries: (1.0 + 0.5) / 2 = 0.75
    mrr = mean_reciprocal_rank(
        [["c1", "c2"], ["c2", "c1"]],
        [{"c1"}, {"c1"}],
    )
    assert mrr == 0.75


def test_dcg_and_ndcg_hand_computed():
    # Graded relevance scores: [3.0, 2.0, 0.0, 1.0]
    # At rank 1: (2^3 - 1) / log2(2) = 7 / 1 = 7.0
    # At rank 2: (2^2 - 1) / log2(3) = 3 / 1.5849625 = 1.892789
    # At rank 3: (2^0 - 1) / log2(4) = 0 / 2 = 0.0
    # At rank 4: (2^1 - 1) / log2(5) = 1 / 2.321928 = 0.430676
    # Sum DCG@4 = 7.0 + 1.892789 + 0.0 + 0.430676 = 9.323465
    scores = [3.0, 2.0, 0.0, 1.0]
    expected_dcg = (
        (7.0 / math.log2(2)) +
        (3.0 / math.log2(3)) +
        (0.0 / math.log2(4)) +
        (1.0 / math.log2(5))
    )
    assert dcg_at_k(scores, k=4) == pytest.approx(expected_dcg, rel=1e-5)

    # NDCG:
    # If retrieved matches ideal order [3.0, 2.0, 1.0, 0.0], NDCG = 1.0
    gt = {"c1": 3.0, "c2": 2.0, "c3": 1.0, "c4": 0.0}
    perfect_retrieved = ["c1", "c2", "c3", "c4"]
    assert ndcg_at_k(perfect_retrieved, gt, k=4) == pytest.approx(1.0)

    # Suboptimal ranking
    suboptimal_retrieved = ["c4", "c3", "c2", "c1"]
    ndcg_sub = ndcg_at_k(suboptimal_retrieved, gt, k=4)
    assert 0.0 < ndcg_sub < 1.0
