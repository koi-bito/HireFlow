"""
pipeline/graph.py
-----------------
LangGraph SequentialAgent - wires all four HireFlow agents into a
StateGraph that executes them in order:

  jd_parser → candidate_matcher → outreach_composer → interview_scheduler

Call `pipeline.invoke(initial_state)` to run the full workflow.
"""

from langgraph.graph import StateGraph, END

from pipeline.state import HireFlowState
from agents.jd_parser import run_jd_parser
from agents.candidate_matcher import run_candidate_matcher
from agents.outreach_composer import run_outreach_composer
from agents.interview_scheduler import run_interview_scheduler


def build_pipeline() -> StateGraph:
    """Construct and compile the HireFlow LangGraph pipeline."""
    graph = StateGraph(HireFlowState)

    # Register each agent as a graph node
    graph.add_node("jd_parser",           run_jd_parser)
    graph.add_node("candidate_matcher",   run_candidate_matcher)
    graph.add_node("outreach_composer",   run_outreach_composer)
    graph.add_node("interview_scheduler", run_interview_scheduler)

    # Define the sequential execution order
    graph.set_entry_point("jd_parser")
    graph.add_edge("jd_parser",           "candidate_matcher")
    graph.add_edge("candidate_matcher",   "outreach_composer")
    graph.add_edge("outreach_composer",   "interview_scheduler")
    graph.add_edge("interview_scheduler", END)

    return graph.compile()


# Module-level compiled pipeline - import and call .invoke() directly
pipeline = build_pipeline()
