"""
evals/metrics.py
----------------
Ranking, Information Retrieval, and Relevance Evaluation Metrics:
  - Precision@k
  - Recall@k
  - Reciprocal Rank (RR) and Mean Reciprocal Rank (MRR)
  - Discounted Cumulative Gain (DCG@k)
  - Normalized Discounted Cumulative Gain (NDCG@k)

All functions are pure, deterministic, and tested against hand-computed values.
"""

import math
from typing import Sequence


def precision_at_k(retrieved: Sequence[str], relevant_set: set[str], k: int) -> float:
    """
    Calculates Precision at rank k.
    Fraction of the top-k retrieved items that are relevant (in relevant_set).
    """
    if k <= 0 or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    num_relevant = sum(1 for item in top_k if item in relevant_set)
    return num_relevant / len(top_k)


def recall_at_k(retrieved: Sequence[str], relevant_set: set[str], k: int) -> float:
    """
    Calculates Recall at rank k.
    Fraction of all relevant items that were successfully retrieved in the top-k.
    """
    if not relevant_set:
        return 1.0 if not retrieved else 0.0
    if k <= 0 or not retrieved:
        return 0.0
    top_k = set(retrieved[:k])
    num_retrieved_relevant = len(top_k & relevant_set)
    return num_retrieved_relevant / len(relevant_set)


def reciprocal_rank(retrieved: Sequence[str], relevant_set: set[str]) -> float:
    """
    Calculates Reciprocal Rank (RR).
    1 / rank of the first relevant item retrieved (1-indexed). Returns 0.0 if no relevant items retrieved.
    """
    for rank_idx, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / rank_idx
    return 0.0


def mean_reciprocal_rank(
    all_retrieved: Sequence[Sequence[str]],
    all_relevant: Sequence[set[str]],
) -> float:
    """
    Calculates Mean Reciprocal Rank (MRR) across a collection of queries.
    """
    if not all_retrieved or not all_relevant or len(all_retrieved) != len(all_relevant):
        return 0.0
    total_rr = sum(reciprocal_rank(ret, rel) for ret, rel in zip(all_retrieved, all_relevant))
    return total_rr / len(all_retrieved)


def dcg_at_k(relevance_scores: Sequence[float], k: int) -> float:
    """
    Computes Discounted Cumulative Gain at rank k using standard graded formula:
    DCG@k = sum_{i=1}^k (2^{rel_i} - 1) / log2(i + 1)
    """
    if k <= 0 or not relevance_scores:
        return 0.0
    dcg = 0.0
    for idx, rel in enumerate(relevance_scores[:k], start=1):
        gain = (2.0 ** rel) - 1.0
        discount = math.log2(idx + 1)
        dcg += gain / discount
    return dcg


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    ground_truth_relevance: dict[str, float],
    k: int,
) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at rank k (NDCG@k).
    Compares DCG of retrieved order to Ideal DCG (IDCG) sorted by ground truth.
    Returns float in [0.0, 1.0].
    """
    if k <= 0 or not retrieved_ids or not ground_truth_relevance:
        return 0.0

    actual_scores = [ground_truth_relevance.get(item, 0.0) for item in retrieved_ids[:k]]
    actual_dcg = dcg_at_k(actual_scores, k)

    # Ideal ranking: sort all available candidates by true relevance descending
    ideal_scores = sorted(ground_truth_relevance.values(), reverse=True)
    ideal_dcg = dcg_at_k(ideal_scores, k)

    if ideal_dcg == 0.0:
        return 1.0 if actual_dcg == 0.0 else 0.0

    return actual_dcg / ideal_dcg
