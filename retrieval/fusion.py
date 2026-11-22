"""
retrieval/fusion.py
-------------------
Pure functions for rank and score fusion (Reciprocal Rank Fusion, min-max scaling).
"""



def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> dict[str, float]:
    """
    Computes Reciprocal Rank Fusion (RRF) scores for items across multiple ranked lists.

    Args:
        ranked_lists: List of rankings, each ranking being an ordered list of item identifiers (e.g. emails).
                      Earlier items in each list have better (lower) rank (0-indexed rank 0 = 1st place).
        k: Smoothing constant to control the impact of high-ranking items (standard literature default: 60).

    Returns:
        Dict mapping each item identifier to its normalized RRF score in [0.0, 1.0].
    """
    if not ranked_lists:
        return {}

    raw_scores: dict[str, float] = {}

    for ranking in ranked_lists:
        for rank_0, item_id in enumerate(ranking):
            rank_1 = rank_0 + 1  # 1-based rank
            rrf_val = 1.0 / (k + rank_1)
            raw_scores[item_id] = raw_scores.get(item_id, 0.0) + rrf_val

    if not raw_scores:
        return {}

    # Normalize to [0.0, 1.0] by dividing by max possible single-item score or max score present
    max_score = max(raw_scores.values())
    if max_score > 0.0:
        return {item: score / max_score for item, score in raw_scores.items()}

    return {item: 0.0 for item in raw_scores}


def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Scales a dictionary of raw float scores to the [0.0, 1.0] interval."""
    if not scores:
        return {}
    values = list(scores.values())
    min_val, max_val = min(values), max(values)
    if min_val == max_val:
        return {k: 1.0 if max_val > 0 else 0.0 for k in scores}
    denom = max_val - min_val
    return {k: (v - min_val) / denom for k, v in scores.items()}
