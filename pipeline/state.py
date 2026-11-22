"""
pipeline/state.py
-----------------
Shared state schema for the HireFlow multi-agent pipeline.

Every agent reads from and writes to this TypedDict. LangGraph's StateGraph
passes it between nodes; each agent function receives the full state and
returns only the keys it modifies.
"""

from typing import Optional, TypedDict


class HireFlowState(TypedDict):
    # ── Input ─────────────────────────────────────────────────────────────
    raw_jd: str                        # raw job description text
    candidates_csv_path: str           # path to the candidate CSV file

    # ── Agent 1 output ────────────────────────────────────────────────────
    parsed_jd: Optional[dict]          # structured JD: role, skills, exp, location, salary

    # ── Agent 2 output ────────────────────────────────────────────────────
    ranked_candidates: Optional[list]  # list of dicts: name, score, match_reasons

    # ── Agent 3 output ────────────────────────────────────────────────────
    outreach_messages: Optional[list]  # list of dicts: candidate_name, subject, body

    # ── Agent 4 output ────────────────────────────────────────────────────
    schedule: Optional[list]           # list of dicts: candidate_name, slots, notes

    # ── Meta ──────────────────────────────────────────────────────────────
    errors: list                       # per-agent error messages for debugging
