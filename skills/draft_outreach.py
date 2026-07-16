"""
Skill: Draft Candidate Outreach
Generates personalized outreach messages per candidate.
"""
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def draft_outreach_message(candidate_name: str, candidate_skills_json: str,
                            role_title: str, company_name: str,
                            fit_reason: str) -> dict:
    """
    Generates a personalized recruiter outreach email for a specific candidate.
    References the candidate's actual skills and explains why they were shortlisted.
    Use this for each candidate after ranking is complete.

    Args:
        candidate_name: Full name of the candidate.
        candidate_skills_json: JSON string of the candidate's skills list.
        role_title: Title of the open position.
        company_name: Name of the hiring company.
        fit_reason: Why this candidate fits (from the scorer output).

    Returns:
        Dict with 'subject' (email subject) and 'message' (email body).
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    skills = json.loads(candidate_skills_json)

    prompt = f"""Write a personalized recruiter outreach email.

Candidate: {candidate_name}
Skills: {', '.join(skills)}
Role: {role_title}
Company: {company_name}
Why they fit: {fit_reason}

Rules:
- Under 150 words
- Reference 2-3 of their specific skills by name
- Sound human, not corporate
- No "I came across your profile" or "exciting opportunity"
- End with a soft ask (open to a quick chat?) not a formal interview request

Respond ONLY with valid JSON. No markdown, no backticks.

Return:
{{
  "subject": "email subject here",
  "message": "full email body here"
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
            "subject": f"{role_title} — Would love to connect",
            "message": f"Hi {candidate_name}, your background in {skills[0] if skills else 'your field'} looks like a strong fit for our {role_title} role. Open to a quick chat?",
            "error": str(e)
        }
