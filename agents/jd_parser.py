"""
agents/jd_parser.py
--------------------
Agent 1 - JD Parser

Receives raw job description text from state["raw_jd"], prompts Gemini to
extract structured requirements, and writes the result to state["parsed_jd"].

If the LLM response cannot be parsed as JSON, a safe default is written and
the error is appended to state["errors"] so the pipeline continues gracefully.
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

# ── Gemini client setup ────────────────────────────────────────────────────────
_API_KEY = os.getenv("GOOGLE_API_KEY")
_MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.5-flash")

_client = None


def get_genai_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    return _client

# ── Prompt template ────────────────────────────────────────────────────────────
_PROMPT_TEMPLATE = """You are a recruitment data extractor. Given the job description below, extract the following fields and return ONLY valid JSON with no other text, markdown, or explanation:

{{
  "job_title": "string - the role title",
  "required_skills": ["list", "of", "must-have", "skills"],
  "preferred_skills": ["list", "of", "nice-to-have", "skills"],
  "years_of_experience": "string e.g. '3-5 years'",
  "location": "string - city/remote/etc",
  "salary_range": "string - if mentioned, else null",
  "employment_type": "string - full-time/part-time/contract/etc"
}}

Job Description:
{raw_jd}
"""

# ── Safe defaults ──────────────────────────────────────────────────────────────
_SAFE_DEFAULT: dict = {
    "job_title": "Unknown",
    "required_skills": [],
    "preferred_skills": [],
    "years_of_experience": "Not specified",
    "location": "Not specified",
    "salary_range": None,
    "employment_type": "Not specified",
}


def _extract_json(text: str) -> dict:
    """Try to extract a JSON object from the model's response text."""
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip()
    return json.loads(text)


def run_jd_parser(state: HireFlowState) -> dict:
    """
    LangGraph node function for Agent 1 - JD Parser.

    Args:
        state: The current HireFlowState.

    Returns:
        A dict with updated keys: 'parsed_jd' and (optionally) 'errors'.
    """
    raw_jd: str = state.get("raw_jd", "").strip()
    errors: list = list(state.get("errors", []))

    if not raw_jd:
        errors.append("JD Parser: raw_jd is empty - skipping LLM call.")
        logger.warning("JD Parser: raw_jd is empty.")
        return {"parsed_jd": _SAFE_DEFAULT.copy(), "errors": errors}

    prompt = _PROMPT_TEMPLATE.format(raw_jd=raw_jd)

    try:
        client = _client if _client is not None else get_genai_client()
        response = client.models.generate_content(model=_MODEL_NAME, contents=prompt)
        parsed = _extract_json(response.text)

        # Guarantee required_skills is always a list
        if not isinstance(parsed.get("required_skills"), list):
            parsed["required_skills"] = []
        if not isinstance(parsed.get("preferred_skills"), list):
            parsed["preferred_skills"] = []

        logger.info("JD Parser: successfully parsed JD for role '%s'.", parsed.get("job_title"))
        return {"parsed_jd": parsed, "errors": errors}

    except json.JSONDecodeError as exc:
        msg = f"JD Parser: JSON decode error - {exc}"
        logger.error(msg)
        errors.append(msg)
        return {"parsed_jd": _SAFE_DEFAULT.copy(), "errors": errors}

    except Exception as exc:  # noqa: BLE001
        msg = f"JD Parser: unexpected error - {exc}"
        logger.exception(msg)
        errors.append(msg)
        return {"parsed_jd": _SAFE_DEFAULT.copy(), "errors": errors}
