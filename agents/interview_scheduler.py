"""
agents/interview_scheduler.py
------------------------------
Agent 4 - Interview Scheduler

Reads state["ranked_candidates"] (specifically strong_match and partial_match
candidates) and generates 3 structured interview slot options per candidate
across the next 5 business days.

Dates are computed deterministically with Python datetime - the LLM is NOT
used for date logic to avoid hallucinations.  The LLM is optionally used to
generate a brief personalised scheduling note per candidate.

Writes results to state["schedule"].
"""

import logging
import os
from datetime import date, timedelta

from dotenv import load_dotenv
from google import genai

from pipeline.state import HireFlowState

load_dotenv()
logger = logging.getLogger(__name__)

_API_KEY = os.getenv("GOOGLE_API_KEY")
_MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.5-flash")
_SLOT_TIMES = ["10:00 AM", "2:00 PM", "11:00 AM"]
_SLOT_DURATION_MIN = 45
_INTERVIEW_FLAGS = {"strong_match", "partial_match"}

_client = genai.Client(api_key=_API_KEY)

_NOTE_PROMPT = """Write a single short scheduling note (max 20 words) for a recruiter about interviewing this candidate. Return ONLY the note as plain text, no JSON, no formatting.

Candidate: {name}
Flag: {flag}
Match reasons: {reasons}
"""


def _next_n_business_days(n: int = 5, start: date | None = None) -> list[date]:
    """Return the next n weekday dates starting from the day after start."""
    if start is None:
        start = date.today()
    days = []
    current = start + timedelta(days=1)
    while len(days) < n:
        if current.weekday() < 5:  # Mon–Fri
            days.append(current)
        current += timedelta(days=1)
    return days


def _generate_slots(candidate_index: int) -> list[dict]:
    """
    Generate 3 interview slots for one candidate.
    Different candidates get offset start days to avoid all slots clustering on day 1.
    """
    offset = candidate_index  # first candidate starts day 1, second day 2, etc.
    business_days = _next_n_business_days(n=5 + offset)
    selected_days = business_days[offset: offset + 3]

    slots = []
    for i, day in enumerate(selected_days):
        slots.append({
            "date": day.strftime("%Y-%m-%d"),
            "time": _SLOT_TIMES[i % len(_SLOT_TIMES)],
            "duration_min": _SLOT_DURATION_MIN,
        })
    return slots


def _generate_note(candidate: dict) -> str:
    """Ask Gemini for a brief scheduling note. Returns a safe default on failure."""
    prompt = _NOTE_PROMPT.format(
        name=candidate["name"],
        flag=candidate.get("flag", "partial_match"),
        reasons="; ".join(candidate.get("match_reasons", [])[:2]),
    )
    try:
        response = _client.models.generate_content(model=_MODEL_NAME, contents=prompt)
        return response.text.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Interview Scheduler: note generation failed for %s - %s", candidate["name"], exc)
        flag = candidate.get("flag", "partial_match")
        return "Strong candidate - prioritise scheduling." if flag == "strong_match" else "Good fit - schedule at convenience."


def run_interview_scheduler(state: HireFlowState) -> dict:
    """
    LangGraph node function for Agent 4 - Interview Scheduler.

    Args:
        state: The current HireFlowState.

    Returns:
        Updated keys: 'schedule' and (optionally) 'errors'.
    """
    errors: list = list(state.get("errors", []))
    ranked_candidates: list = state.get("ranked_candidates") or []

    # Only schedule for strong_match and partial_match candidates
    interview_candidates = [
        c for c in ranked_candidates if c.get("flag") in _INTERVIEW_FLAGS
    ]

    if not interview_candidates:
        errors.append("Interview Scheduler: no suitable candidates to schedule.")
        logger.warning("Interview Scheduler: no strong/partial match candidates found.")
        return {"schedule": [], "errors": errors}

    schedule = []
    for idx, candidate in enumerate(interview_candidates):
        slots = _generate_slots(idx)
        note = _generate_note(candidate)
        schedule.append({
            "candidate_name": candidate["name"],
            "email": candidate.get("email", ""),
            "flag": candidate.get("flag"),
            "slots": slots,
            "notes": note,
        })
        logger.info("Interview Scheduler: scheduled %s.", candidate["name"])

    return {"schedule": schedule, "errors": errors}
