# HireFlow - Multi-Agent Recruitment System
## Implementation Plan: v3 Embeddings, Hybrid Ranking, Personalisation & Evals

> **Context:** HireFlow was rebuilt from an early prototype into a production-grade multi-agent recruitment platform orchestrated by LangGraph, powered by Google GenAI (Gemini 3.5 Flash), featuring two-tower hybrid retrieval (dense vector embeddings + keyword overlap via Reciprocal Rank Fusion), recruiter feedback memory, and an evaluation harness with NDCG/MRR metrics.

---

## Target Architecture

```
User Input (JD + Candidate Profiles)
        │
        ▼
┌────────────────────────────────────────────────────────┐
│  Stage 1: JD Parser (Agent 1)                          │
│  - Extracts structured role, skills, exp, location     │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Stage 2: Candidate Matcher (Agent 2 - v3 Funnel)      │
│  - Two-Tower Retrieval: Dense Vectors + Keyword Overlap│
│  - Reciprocal Rank Fusion (RRF: k=60)                  │
│  - Soft Boosts: Experience, Location, Personalisation  │
│  - LLM Re-ranking: Deep reasoning on Top 5 candidates  │
│  - Grounded Explainability & Score Blending (0.4 / 0.6)│
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Stage 3: Outreach Composer (Agent 3)                  │
│  - Personalised email per top candidate                │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Stage 4: Interview Scheduler (Agent 4)                │
│  - 3 deterministic business-day slots per candidate    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│  Stage 5: Recruiter Chat Agent (Agent 5)               │
│  - Function-calling loop with conversational memory    │
│  - Tools: search, explain, feedback, outreach, parser  │
└────────────────────────────────────────────────────────┘
```

---

## Progress Tracker

| Phase | Description | Status |
|---|---|---|
| 0 | Repo cleanup, audit & stale test resolution | ✅ Completed |
| 1 | Config management (`config.py`) & dependencies (`requirements.txt`) | ✅ Completed |
| 2 | Embedding layer (`retrieval/embedder.py`), caching & `CandidateIndex` | ✅ Completed |
| 3 | Hybrid retrieval (RRF in `retrieval/fusion.py`), boosts & LLM re-rank | ✅ Completed |
| 4 | SQLite database persistence (`storage/db.py`) & recruiter personalisation | ✅ Completed |
| 5 | Agent tool use (`agents/recruiter_chat.py`) & conversational session memory | ✅ Completed |
| 6 | Evals harness (`evals/run_eval.py`, `evals/metrics.py`) & benchmark dataset | ✅ Completed |
| 7 | Reliability, Docker (`Dockerfile`, `docker-compose.yml`), CI & smoke test | ✅ Completed |
| 8 | Unit & integration test suite (92 offline tests in `tests/`) | ✅ Completed |
| 9 | UTF-8 README, DESIGN document (`docs/DESIGN.md`) & LEARNINGS update | ✅ Completed |
