"""
Integration tests for HireFlow agents.
Requires GOOGLE_API_KEY in .env
Run: pytest tests/test_agents.py -v
"""
import json
import os
import sys
import asyncio
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


def run_agent(agent, message_text: str) -> str:
    """Helper: runs an agent and returns its final response text."""
    session_service = InMemorySessionService()
    # Apply the asyncio fix for create_session here as well!
    session = asyncio.run(session_service.create_session(app_name="test", user_id="test"))
    
    runner = Runner(agent=agent, app_name="test", session_service=session_service)
    message = types.Content(role="user", parts=[types.Part(text=message_text)])
    for event in runner.run(user_id="test", session_id=session.id, new_message=message):
        if event.is_final_response():
            return event.response.parts[0].text
    return ""


def test_jd_analyst_extracts_role():
    from agents.jd_analyst.agent import jd_analyst_agent
    jd = "We are hiring a Senior Python Developer with 5+ years experience in ML."
    response = run_agent(jd_analyst_agent, f"Analyze this JD: {jd}")
    assert "python" in response.lower() or "developer" in response.lower()


def test_jd_analyst_flags_bias():
    from agents.jd_analyst.agent import jd_analyst_agent
    jd = "Looking for a young rockstar. Must be a native English speaker."
    response = run_agent(jd_analyst_agent, f"Analyze this JD: {jd}")
    bias_words = ["bias", "young", "native", "rockstar", "flag", "discriminat"]
    assert any(w in response.lower() for w in bias_words)


def test_matcher_returns_ranking():
    from agents.matcher.agent import matcher_agent
    jd = "Python machine learning engineer with scikit-learn experience."
    candidates = [
        {"name": "Alice", "skills": ["Python", "machine learning", "scikit-learn"]},
        {"name": "Bob", "skills": ["Java", "Spring"]}
    ]
    msg = f"Rank these:\nJD: {jd}\nCandidates: {json.dumps(candidates)}"
    response = run_agent(matcher_agent, msg)
    assert "alice" in response.lower()
    assert "bob" in response.lower()


def test_outreach_mentions_candidate_name():
    from agents.outreach.agent import outreach_agent
    msg = """Draft outreach for this candidate:
    Name: Alice Sharma
    Skills: ["Python", "machine learning", "NLP"]
    Role: ML Engineer
    Company: TechCorp
    Fit reason: Strong Python and NLP background matches role requirements."""
    response = run_agent(outreach_agent, msg)
    assert "alice" in response.lower()
