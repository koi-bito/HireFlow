"""
tests/test_chat.py
-------------------
Unit tests for Agent 5 - Recruiter Chatbot (tool use, memory retention, history compaction).
"""

from unittest.mock import MagicMock, patch

from agents.recruiter_chat import (
    _SESSIONS,
    _compact_session_history,
    handle_recruiter_chat,
    tool_draft_outreach,
    tool_explain_match,
    tool_parse_job_description,
    tool_record_feedback,
    tool_search_candidates,
)


def test_tool_parse_job_description():
    with patch("agents.jd_parser._client") as mock_client:
        mock_resp = MagicMock()
        mock_resp.text = '{"job_title": "Data Scientist", "required_skills": ["Python", "Pandas"]}'
        mock_client.models.generate_content.return_value = mock_resp

        parsed = tool_parse_job_description("Hiring Data Scientist with Python and Pandas.")
        assert parsed["job_title"] == "Data Scientist"
        assert "Python" in parsed["required_skills"]


def test_tool_search_candidates():
    with patch("agents.candidate_matcher._client") as mock_client:
        mock_resp = MagicMock()
        mock_resp.text = '{"score": 85, "match_reasons": ["Python", "ML"], "flag": "strong_match"}'
        mock_client.models.generate_content.return_value = mock_resp

        results = tool_search_candidates(job_title="ML Engineer", required_skills=["Python"], top_k=2)
        assert len(results) <= 2
        assert all("email" in c and "score" in c for c in results)


def test_tool_explain_match():
    res = tool_explain_match("alice@example.com")
    assert res["name"] == "Alice Sharma"
    assert "FastAPI" in res["skills"]

    # Not found candidate
    bad = tool_explain_match("ghost@nowhere.com")
    assert "error" in bad


def test_tool_record_feedback():
    res = tool_record_feedback("alice@example.com", "Senior ML Engineer", "shortlisted")
    assert res["status"] == "success"
    assert res["recorded"]["action"] == "shortlisted"


def test_tool_draft_outreach():
    with patch("agents.outreach_composer._client") as mock_client:
        mock_resp = MagicMock()
        mock_resp.text = '{"subject": "ML Role at HireFlow", "body": "Hi Alice, we love your work!"}'
        mock_client.models.generate_content.return_value = mock_resp

        outreach = tool_draft_outreach("Alice Sharma", "alice@example.com", "ML Engineer")
        assert "subject" in outreach
        assert "body" in outreach


def test_chat_turn_and_memory_retention():
    session_id = "test-session-mem-1"
    # Turn 1
    r1 = handle_recruiter_chat("Find candidates for ML Engineer", session_id=session_id)
    assert r1["session_id"] == session_id
    assert len(_SESSIONS[session_id]) == 2  # user + assistant

    # Turn 2: carries session
    r2 = handle_recruiter_chat("Shortlist candidate alice@example.com for this role", session_id=session_id)
    assert len(_SESSIONS[session_id]) == 4  # 2 user + 2 assistant
    assert any(tc["tool"] == "record_feedback" for tc in r2["tool_calls"])


def test_session_compaction():
    # Long history of 15 turns
    history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"msg {i}"} for i in range(15)]
    compacted = _compact_session_history(history)
    assert len(compacted) <= 7
    assert compacted[0]["role"] == "system"
    assert "Summary" in compacted[0]["content"]
