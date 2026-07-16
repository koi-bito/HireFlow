"""
HireFlow Pipeline Runner
Run: python run_pipeline.py
"""
import os
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from agents.orchestrator.agent import hireflow_pipeline

# Load candidates from CSV
df = pd.read_csv("data/sample_candidates.csv")
candidates = [
    {
        "name": row["name"],
        "skills": [s.strip() for s in row["skills"].split(",")]
    }
    for _, row in df.iterrows()
]

# Sample JD
jd = """
Senior Machine Learning Engineer

We are looking for an ML Engineer to join our AI team.
Required: Python, PyTorch or TensorFlow, model training and deployment experience.
Nice to have: NLP, computer vision, MLflow, Docker.
3+ years of professional ML experience required.
"""

input_text = f"""Process this recruitment request through the full pipeline.

JOB DESCRIPTION:
{jd}

CANDIDATES:
{json.dumps(candidates, indent=2)}

Run JD analysis, bias check, rank the candidates, then draft outreach for the top 3."""

# Set up runner
session_service = InMemorySessionService()
runner = Runner(
    agent=hireflow_pipeline,
    app_name="hireflow",
    session_service=session_service
)
import asyncio
session = asyncio.run(session_service.create_session(app_name="hireflow", user_id="recruiter"))
message = types.Content(role="user", parts=[types.Part(text=input_text)])

print("Running HireFlow pipeline...\n")
print("=" * 60)

for event in runner.run(user_id="recruiter", session_id=session.id, new_message=message):
    if event.is_final_response():
        print(event.response.parts[0].text)

print("=" * 60)
print("\nPipeline complete.")
