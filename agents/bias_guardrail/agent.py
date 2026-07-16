from google.adk.agents import LlmAgent
from skills.detect_bias import detect_bias_in_jd

bias_guardrail_agent = LlmAgent(
    name="bias_guardrail_agent",
    model="gemini-2.0-flash",
    description=(
        "A compliance agent that audits job descriptions for bias and "
        "discrimination. Always run this before finalizing any recruitment output."
    ),
    instruction="""You are an HR compliance officer focused on fair hiring.

When given a JD:
- Run detect_bias_in_jd
- severity 'none' or 'low': confirm clean with brief notes
- severity 'medium': list each flag with specific rewrite suggestions
- severity 'high': clearly warn this JD must NOT be published as-is

Be specific. Point to exact phrases. Generic advice is useless.""",
    tools=[detect_bias_in_jd]
)
