"""
Skill: Parse Job Description
"""
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def parse_job_description(jd_text: str) -> dict:
    """
    Parses a raw job description and extracts structured recruitment data:
    role title, required skills, preferred skills, seniority level, and role type.
    Use this first, before any candidate matching or sourcing.

    Args:
        jd_text: The full text of the job description.

    Returns:
        Dict with role_title, required_skills, preferred_skills,
        experience_level, role_type, and summary.
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""Extract structured information from this job description.
Respond ONLY with valid JSON. No markdown, no backticks, no explanation.

Job Description:
{jd_text}

Return exactly this structure:
{{
  "role_title": "string",
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["skill1", "skill2"],
  "experience_level": "junior|mid|senior",
  "role_type": "technical|non-technical",
  "summary": "one sentence summary"
}}"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        return {
            "error": str(e),
            "role_title": "Unknown",
            "required_skills": [],
            "preferred_skills": [],
            "experience_level": "unknown",
            "role_type": "unknown",
            "summary": "Failed to parse JD"
        }
