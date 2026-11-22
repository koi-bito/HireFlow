"""
tests/test_personalisation.py
------------------------------
Unit tests for recruiter feedback personalisation nudges.
Tests that shortlisted feedback produces positive nudges and rejected feedback produces negative nudges.
"""


from retrieval.personalisation import calculate_personalisation_nudge


def test_no_feedback_returns_zero_nudge():
    nudge, note = calculate_personalisation_nudge(
        candidate_email="alice@test.com",
        candidate_profile_text="Python ML",
        candidate_vector=None,
        past_feedback=[],
    )
    assert nudge == 0.0
    assert note is None


def test_shortlisted_feedback_boosts_candidate():
    past_feedback = [
        {"candidate_email": "alice@test.com", "action": "shortlisted", "job_title": "ML Engineer"},
    ]
    nudge, note = calculate_personalisation_nudge(
        candidate_email="alice@test.com",
        candidate_profile_text="Python ML",
        candidate_vector=None,
        past_feedback=past_feedback,
    )
    assert nudge > 0.0
    assert "positive" in note.lower()


def test_rejected_feedback_demotes_candidate():
    past_feedback = [
        {"candidate_email": "bob@test.com", "action": "rejected", "job_title": "ML Engineer"},
    ]
    nudge, note = calculate_personalisation_nudge(
        candidate_email="bob@test.com",
        candidate_profile_text="Java Backend",
        candidate_vector=None,
        past_feedback=past_feedback,
    )
    assert nudge < 0.0
    assert "rejected" in note.lower()


def test_nudge_is_strictly_capped():
    # Multiple feedback items
    past_feedback = [
        {"candidate_email": "alice@test.com", "action": "hired"},
        {"candidate_email": "alice@test.com", "action": "shortlisted"},
    ]
    nudge, note = calculate_personalisation_nudge(
        candidate_email="alice@test.com",
        candidate_profile_text="Python ML",
        candidate_vector=None,
        past_feedback=past_feedback,
        max_nudge=0.08,
    )
    assert -0.08 <= nudge <= 0.08
