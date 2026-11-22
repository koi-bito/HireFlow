"""
agents/outreach_composer.py
----------------------------
Agent 3 - Outreach Composer

Reads state["ranked_candidates"] and state["parsed_jd"], then prompts
Gemini to write a personalised outreach email for each of the top N
candidates.  Each email references specific match reasons to feel
human rather than templated.

Writes results to state["outreach_messages"].
"""

import json
import logging
import os
import re

from dotenv import load_dotenv
from google import genai

from pipeline.state import HireFlowState

load_dotenv()
logger = logging.getLogger(__name__)

_API_KEY = os.getenv("GOOGLE_API_KEY")
_MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.5-flash")
_MAX_CANDIDATES = int(os.getenv("OUTREACH_MAX_CANDIDATES", "5"))

_client = None


def get_genai_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client

_OUTREACH_PROMPT = """Write a professional recruiter outreach email for this candidate. The tone should be warm and human - not a generic template blast. Reference at least one of their specific strengths listed below. Keep the email under 200 words. Return ONLY valid JSON with no other text or markdown:
{{
  "subject": "email subject line",
  "body": "full email body"
}}

Role: {job_title}
Candidate name: {name}
Why they're a great match:
{match_reasons}
"""


def _parse_llm_response(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    return json.loads(text)


def _compose_outreach(candidate: dict, job_title: str) -> dict:
    """Prompt Gemini to write one outreach email. Returns subject + body dict."""
    reasons_text = "\n".join(f"  - {r}" for r in candidate.get("match_reasons", []))
    prompt = _OUTREACH_PROMPT.format(
        job_title=job_title,
        name=candidate["name"],
        match_reasons=reasons_text or "  - Strong overall profile",
    )
    try:
        client = _client if _client is not None else get_genai_client()
        response = client.models.generate_content(model=_MODEL_NAME, contents=prompt)
        result = _parse_llm_response(response.text)
        return {
            "subject": result.get("subject", "Exciting opportunity for you"),
            "body": result.get("body", ""),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Outreach Composer: LLM call failed for %s - %s", candidate["name"], exc)
        return {
            "subject": f"Opportunity: {job_title}",
            "body": (
                f"Hi {candidate['name']},\n\n"
                f"We came across your profile and believe you could be a great fit for our {job_title} role. "
                "We'd love to connect and share more details.\n\nBest regards,\nThe Recruitment Team"
            ),
        }


def run_outreach_composer(state: HireFlowState) -> dict:
    """
    LangGraph node function for Agent 3 - Outreach Composer.

    Args:
        state: The current HireFlowState.

    Returns:
        Updated keys: 'outreach_messages' and (optionally) 'errors'.
    """
    errors: list = list(state.get("errors", []))
    ranked_candidates: list = state.get("ranked_candidates") or []
    parsed_jd: dict = state.get("parsed_jd") or {}
    job_title: str = parsed_jd.get("job_title", "Open Position")

    if not ranked_candidates:
        errors.append("Outreach Composer: no ranked candidates found - skipping.")
        logger.warning("Outreach Composer: ranked_candidates is empty.")
        return {"outreach_messages": [], "errors": errors}

    top_candidates = ranked_candidates[:_MAX_CANDIDATES]
    outreach_messages = []

    for candidate in top_candidates:
        outreach = _compose_outreach(candidate, job_title)
        outreach_messages.append({
            "candidate_name": candidate["name"],
            "email": candidate.get("email", ""),
            "subject": outreach["subject"],
            "body": outreach["body"],
        })
        logger.info("Outreach Composer: composed email for %s.", candidate["name"])

    return {"outreach_messages": outreach_messages, "errors": errors}
