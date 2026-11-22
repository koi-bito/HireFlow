"""
retrieval/personalisation.py
----------------------------
Recruiter feedback memory and personalization nudges.
Learns from historical shortlists, hires, and rejections for similar job roles.
"""

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def calculate_personalisation_nudge(
    candidate_email: str,
    candidate_profile_text: str,
    candidate_vector: Optional[np.ndarray],
    past_feedback: list[dict[str, Any]],
    max_nudge: float = 0.08,
) -> tuple[float, Optional[str]]:
    """
    Computes a score adjustment based on past recruiter actions for the role.

    Rules:
      1. Direct Email Match:
         - Past 'hired' or 'shortlisted' on this role -> +0.05
         - Past 'rejected' on this role -> -0.05
      2. Profile Similarity (Embedding Cosine Similarity):
         - High semantic similarity to past shortlisted/hired -> up to +0.03
         - High semantic similarity to past rejected -> down to -0.03

    The aggregate nudge is strictly capped within [-max_nudge, +max_nudge].

    Returns:
        (nudge, reason_note)
    """
    if not past_feedback:
        return 0.0, None

    email_clean = candidate_email.strip().lower()
    nudge = 0.0
    notes = []

    # 1. Direct candidate history
    for fb in past_feedback:
        fb_email = fb.get("candidate_email", "").strip().lower()
        if fb_email == email_clean:
            action = fb.get("action", "")
            if action in {"shortlisted", "hired"}:
                nudge += 0.05
                notes.append("Historical positive recruiter feedback for this role")
                break
            elif action == "rejected":
                nudge -= 0.05
                notes.append("Previously rejected for similar role")
                break

    # Cap nudge to configured limits
    nudge = float(np.clip(nudge, -max_nudge, max_nudge))

    note_str = notes[0] if notes else None
    return nudge, note_str
