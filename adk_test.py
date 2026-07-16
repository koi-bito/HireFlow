import os
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# A simple tool
def get_candidate_count() -> dict:
    """Returns the number of candidates in the system."""
    return {"count": 42}

# An agent with that tool
agent = LlmAgent(
    name="test_agent",
    model="gemini-2.0-flash",
    description="A test agent.",
    instruction="Answer questions using your tools.",
    tools=[get_candidate_count]
)

# Runner + session
session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="test", session_service=session_service)
session = session_service.create_session(app_name="test", user_id="kunal")

# Run it
message = types.Content(role="user", parts=[types.Part(text="How many candidates do we have?")])
for event in runner.run(user_id="kunal", session_id=session.id, new_message=message):
    if event.is_final_response():
        print("Response:", event.response.parts[0].text)
