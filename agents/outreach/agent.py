from google.adk.agents import LlmAgent
from skills.draft_outreach import draft_outreach_message

outreach_agent = LlmAgent(
    name="outreach_agent",
    model="gemini-2.0-flash",
    description=(
        "Generates personalized outreach emails for shortlisted candidates. "
        "Each message references the candidate's specific skills and explains "
        "why they were chosen. Use after ranking is complete."
    ),
    instruction="""You are an expert recruitment copywriter.

When given a shortlist and role details:
1. Call draft_outreach_message for each candidate in the shortlist
2. Present each message with the candidate name as a clear header
3. If a candidate has no skills listed, note that more profile data is needed

Your output is what a recruiter will send. Make it count.""",
    tools=[draft_outreach_message]
)
