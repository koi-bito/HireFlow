"""
agents/candidate_matcher.py
----------------------------
Agent 2 - Candidate Matcher

Reads the parsed JD from state["parsed_jd"] and a candidates CSV from
state["candidates_csv_path"].  Uses a two-step approach:

  1. Keyword pre-filter  - fast overlap ratio on required_skills vs candidate skills
  2. LLM scoring pass   - Gemini scores overall fit (0-100) with match reasons

Writes a ranked shortlist to state["ranked_candidates"].
"""

import json
import os
import re
import logging
from typing import Any

import pandas as pd
from google import genai
from dotenv import load_dotenv

from pipeline.state import HireFlowState

load_dotenv()
logger = logging.getLogger(__name__)

_API_KEY = os.getenv("GOOGLE_API_KEY")
_MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")
_PRE_FILTER_THRESHOLD = 0.0   # Keep all candidates, let LLM decide
_TOP_N = 10                   # Max candidates to pass through LLM scoring

_client = genai.Client(api_key=_API_KEY)

_SCORING_PROMPT = """You are a senior technical recruiter. Score this candidate's fit for the role from 0 to 100 and give 3 specific match reasons. Return ONLY valid JSON with no other text:
{{
  "score": <integer 0-100>,
  "match_reasons": ["reason 1", "reason 2", "reason 3"],
  "flag": "strong_match" | "partial_match" | "weak_match"
}}

Role requirements:
{parsed_jd}

Candidate profile:
{candidate_text}
"""

_REQUIRED_CSV_COLUMNS = {"name", "email", "skills", "years_experience", "location", "summary"}


def _keyword_overlap(required_skills: list[str], candidate_skills: str) -> float:
    """Simple overlap ratio between required skills and candidate skills string."""
    if not required_skills:
        return 1.0
    req = {s.lower().strip() for s in required_skills}
    cand = {s.lower().strip() for s in candidate_skills.split(",")}
    overlap = len(req & cand)
    return overlap / len(req)


def _build_candidate_text(row: pd.Series) -> str:
    """Format a candidate row as a readable text block for the LLM prompt."""
    return (
        f"Name: {row['name']}\n"
        f"Skills: {row['skills']}\n"
        f"Years of Experience: {row['years_experience']}\n"
        f"Location: {row['location']}\n"
        f"Summary: {row['summary']}"
    )


def _parse_llm_response(text: str) -> dict:
    """Strip markdown fences and parse JSON from LLM response."""
    text = re.sub(r"```(?:json)?", "", text).strip()
    return json.loads(text)


def _score_candidate(candidate_text: str, parsed_jd: dict) -> dict:
    """Call Gemini to score a single candidate. Returns score dict or safe default."""
    prompt = _SCORING_PROMPT.format(
        parsed_jd=json.dumps(parsed_jd, indent=2),
        candidate_text=candidate_text,
    )
    try:
        response = _client.models.generate_content(model=_MODEL_NAME, contents=prompt)
        result = _parse_llm_response(response.text)
        # Ensure required keys exist
        result.setdefault("score", 0)
        result.setdefault("match_reasons", [])
        result.setdefault("flag", "weak_match")
        return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("Candidate Matcher: LLM scoring failed - %s", exc)
        return {"score": 0, "match_reasons": [], "flag": "weak_match"}


def run_candidate_matcher(state: HireFlowState) -> dict:
    """
    LangGraph node function for Agent 2 - Candidate Matcher.

    Args:
        state: The current HireFlowState.

    Returns:
        Updated keys: 'ranked_candidates' and (optionally) 'errors'.
    """
    errors: list = list(state.get("errors", []))
    parsed_jd: dict = state.get("parsed_jd") or {}
    csv_path: str = state.get("candidates_csv_path", "data/sample_candidates.csv")

    # ── Load CSV ───────────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        msg = f"Candidate Matcher: CSV not found at '{csv_path}' - trying default."
        logger.warning(msg)
        errors.append(msg)
        try:
            df = pd.read_csv("data/sample_candidates.csv")
        except Exception as exc:
            err = f"Candidate Matcher: cannot load any candidate CSV - {exc}"
            errors.append(err)
            return {"ranked_candidates": [], "errors": errors}
    except Exception as exc:
        err = f"Candidate Matcher: error reading CSV - {exc}"
        errors.append(err)
        return {"ranked_candidates": [], "errors": errors}

    # ── Validate columns ───────────────────────────────────────────────────────
    missing = _REQUIRED_CSV_COLUMNS - set(df.columns)
    if missing:
        err = f"Candidate Matcher: CSV missing required columns: {missing}"
        errors.append(err)
        logger.error(err)
        return {"ranked_candidates": [], "errors": errors}

    df = df.fillna("")

    # ── Step 1: Keyword pre-filter ─────────────────────────────────────────────
    required_skills: list[str] = parsed_jd.get("required_skills", [])
    df["_overlap"] = df["skills"].apply(
        lambda s: _keyword_overlap(required_skills, s)
    )
    candidates_filtered = (
        df[df["_overlap"] >= _PRE_FILTER_THRESHOLD]
        .sort_values("_overlap", ascending=False)
        .head(_TOP_N)
    )

    # ── Step 2: LLM scoring ────────────────────────────────────────────────────
    ranked: list[dict[str, Any]] = []
    for _, row in candidates_filtered.iterrows():
        candidate_text = _build_candidate_text(row)
        score_result = _score_candidate(candidate_text, parsed_jd)
        ranked.append({
            "name": row["name"],
            "email": row["email"],
            "score": score_result["score"],
            "flag": score_result["flag"],
            "match_reasons": score_result["match_reasons"],
        })

    # Sort by score descending
    ranked.sort(key=lambda x: x["score"], reverse=True)
    logger.info("Candidate Matcher: ranked %d candidates.", len(ranked))
    return {"ranked_candidates": ranked, "errors": errors}
