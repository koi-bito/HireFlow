"""
HireFlow Orchestrator
Coordinates the full recruitment pipeline in sequence.
"""
from google.adk.agents import SequentialAgent
from agents.jd_analyst.agent import jd_analyst_agent
from agents.bias_guardrail.agent import bias_guardrail_agent
from agents.matcher.agent import matcher_agent
from agents.outreach.agent import outreach_agent

hireflow_pipeline = SequentialAgent(
    name="hireflow_pipeline",
    description=(
        "The HireFlow multi-agent recruitment pipeline. Given a job description "
        "and candidate list, runs JD analysis → bias audit → candidate matching "
        "→ outreach drafting in sequence. Returns a complete recruitment package."
    ),
    sub_agents=[
        jd_analyst_agent,
        bias_guardrail_agent,
        matcher_agent,
        outreach_agent
    ]
)
