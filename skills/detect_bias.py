"""
Skill: Bias Detection Guardrail
"""
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

KNOWN_BIAS_PATTERNS = [
    "young", "energetic team", "digital native", "recent graduate only",
    "native english", "brotherhood", "gentleman", "rockstar", "ninja",
    "culture fit", "must be local", "guys"
]


def detect_bias_in_jd(jd_text: str) -> dict:
    """
    Analyzes a job description for discriminatory or biased language.
    Checks for both hardcoded known-bias patterns and uses an LLM to
    detect subtler forms of exclusionary language. Always run this
    before publishing a JD or sending candidate shortlists.

    Args:
        jd_text: The full text of the job description.

    Returns:
        Dict with bias_detected (bool), severity (none/low/medium/high),
        flags (list of issues), and suggestions (list of fixes).
    """
    flags = []
    jd_lower = jd_text.lower()
    for pattern in KNOWN_BIAS_PATTERNS:
        if pattern in jd_lower:
            flags.append(f"Known bias pattern: '{pattern}'")

    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""You are an HR compliance expert. Analyze this job description for bias.

Job Description:
{jd_text}

Check for: age bias, gender bias, nationality bias, disability exclusion,
culture-fit language that excludes minorities, unnecessarily restrictive requirements.

Respond ONLY with valid JSON. No markdown, no backticks.

Return:
{{
  "additional_flags": ["issue1", "issue2"],
  "severity": "none|low|medium|high",
  "suggestions": ["fix1", "fix2"],
  "overall_assessment": "one sentence"
}}"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        llm_result = json.loads(raw.strip())

        all_flags = flags + llm_result.get("additional_flags", [])
        severity = llm_result.get("severity", "low") if all_flags else "none"

        return {
            "bias_detected": len(all_flags) > 0,
            "severity": severity,
            "flags": all_flags,
            "suggestions": llm_result.get("suggestions", []),
            "overall_assessment": llm_result.get("overall_assessment", "No issues found.")
        }
    except Exception as e:
        return {
            "bias_detected": len(flags) > 0,
            "severity": "unknown",
            "flags": flags,
            "suggestions": [],
            "overall_assessment": f"LLM check failed: {str(e)}. Pattern check only.",
            "error": str(e)
        }
