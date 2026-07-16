from google.adk.agents import LlmAgent
from skills.parse_jd import parse_job_description
from skills.detect_bias import detect_bias_in_jd

jd_analyst_agent = LlmAgent(
    name="jd_analyst_agent",
    model="gemini-2.0-flash",
    description=(
        "Analyzes job descriptions. Extracts structured requirements "
        "(skills, seniority, role type) and runs an initial bias check. "
        "Always use this first, before any candidate matching."
    ),
    instruction="""You are a senior HR analyst specializing in job description analysis.

When given a job description:
1. Call parse_job_description to extract structured requirements
2. Call detect_bias_in_jd to check for discriminatory language
3. If bias is detected at medium or high severity, highlight it clearly
4. Summarize the key requirements concisely for the recruiter

Be direct. Recruiters are busy. No filler.""",
    tools=[parse_job_description, detect_bias_in_jd]
)
