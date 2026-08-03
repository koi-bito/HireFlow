# HireFlow — Multi-Agent Recruitment System
## Implementation Plan: Rebuild with Google ADK + LangGraph

> **Context:** HireFlow was originally a Flask + TF-IDF prototype for candidate matching and chatbot simulation. This plan covers the full rebuild into a production multi-agent recruitment system using Google ADK and LangGraph. The target architecture — as described in the portfolio and resume — is a `SequentialAgent` orchestrating four specialised `LlmAgent`s: JD Parser, Candidate Matcher, Outreach Composer, and Interview Scheduler.

---

## Current State (Old Flask Prototype)

```
app.py          — Flask routes + CSV upload handling
matcher.py      — TF-IDF + cosine similarity scoring
chatbot.py      — Rule-based chat simulation
data/           — Sample candidate CSVs
templates/      — HTML UI
static/         — CSS
```

The old code is kept on a `legacy` branch. The `main` branch will hold the new multi-agent rebuild.

---

## Target Architecture

```
User Input (JD + Candidate Profiles)
        │
        ▼
┌───────────────────────┐
│   SequentialAgent     │  ← Google ADK orchestrator
│   (LangGraph graph)   │
└───────────┬───────────┘
            │
    ┌───────┴────────┐
    │  Pipeline Flow │
    └───────┬────────┘
            │
    ┌───────▼────────────┐
    │  Agent 1           │
    │  JD Parser         │  → extracts: role, skills, exp, location, salary
    └───────┬────────────┘
            │
    ┌───────▼────────────┐
    │  Agent 2           │
    │  Candidate Matcher │  → NLP similarity scoring, ranked shortlist
    └───────┬────────────┘
            │
    ┌───────▼────────────┐
    │  Agent 3           │
    │  Outreach Composer │  → personalised email/message per candidate
    └───────┬────────────┘
            │
    ┌───────▼────────────┐
    │  Agent 4           │
    │  Interview         │  → schedules slots, outputs calendar-ready data
    │  Scheduler         │
    └────────────────────┘
            │
            ▼
    Final Output: ranked candidates + outreach drafts + schedule
```

---

## Phase 0 — Repository Cleanup & Setup

**Goal:** Clean slate, proper structure, old code preserved.

### Tasks

- [ ] Create `legacy` branch, push all existing Flask code there
- [ ] Clear `main` branch, set up new project structure:

```
HireFlow/
├── agents/
│   ├── __init__.py
│   ├── jd_parser.py
│   ├── candidate_matcher.py
│   ├── outreach_composer.py
│   └── interview_scheduler.py
├── pipeline/
│   ├── __init__.py
│   ├── graph.py          ← LangGraph SequentialAgent definition
│   └── state.py          ← shared state schema (TypedDict)
├── data/
│   └── sample_candidates.csv
├── tests/
│   ├── test_jd_parser.py
│   ├── test_matcher.py
│   └── test_pipeline.py
├── app.py                ← FastAPI entry point
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── Implementation_Plan.md
```

- [ ] `requirements.txt`:
  ```
  google-generativeai
  google-adk
  langgraph
  langchain
  langchain-google-genai
  fastapi
  uvicorn
  pandas
  python-dotenv
  pytest
  ```

- [ ] `.env.example`:
  ```
  GOOGLE_API_KEY=your_gemini_api_key_here
  MODEL_NAME=gemini-1.5-flash
  ```

---

## Phase 1 — Shared State Schema

**Goal:** Define what data flows between agents. Every agent reads from and writes to this shared state object.

### File: `pipeline/state.py`

```python
from typing import TypedDict, Optional
from dataclasses import dataclass, field

class HireFlowState(TypedDict):
    # Input
    raw_jd: str                          # raw job description text
    candidates_csv_path: str             # path to uploaded candidate CSV

    # Agent 1 output
    parsed_jd: Optional[dict]            # structured JD: role, skills, exp, location, salary

    # Agent 2 output
    ranked_candidates: Optional[list]    # list of dicts: name, score, match_reasons

    # Agent 3 output
    outreach_messages: Optional[list]    # list of dicts: candidate_name, subject, body

    # Agent 4 output
    schedule: Optional[list]             # list of dicts: candidate_name, slots, notes

    # Meta
    errors: list                         # any per-agent error messages
```

**Why this matters:** LangGraph's StateGraph passes this TypedDict between nodes. Each agent function receives the full state, does its work, and returns only the keys it modifies.

---

## Phase 2 — Agent 1: JD Parser

**Goal:** Take raw job description text → structured dictionary of requirements.

### File: `agents/jd_parser.py`

**What it does:**
- Receives `state["raw_jd"]`
- Prompts Gemini to extract: job title, required skills (list), preferred skills (list), years of experience, location, salary range, employment type
- Returns structured JSON written to `state["parsed_jd"]`

**Key implementation notes:**
- Use `google.generativeai` with a structured output prompt
- Instruct the model to return only valid JSON — parse with `json.loads()`, handle failures gracefully
- If parsing fails, write error to `state["errors"]` and return safe defaults so the pipeline doesn't crash

**Sample prompt structure:**
```
You are a recruitment data extractor. Given the job description below, extract the following fields and return ONLY valid JSON with no other text:
{
  "job_title": "...",
  "required_skills": [...],
  "preferred_skills": [...],
  "years_of_experience": "...",
  "location": "...",
  "salary_range": "...",
  "employment_type": "..."
}

JD:
{raw_jd}
```

**Tests to write (`tests/test_jd_parser.py`):**
- Given a realistic JD, check all keys are present in output
- Given a malformed/empty JD, confirm no exception is raised
- Confirm `parsed_jd["required_skills"]` is always a list, never None

---

## Phase 3 — Agent 2: Candidate Matcher

**Goal:** Given `parsed_jd` and a CSV of candidates, return a ranked shortlist with match scores and reasons.

### File: `agents/candidate_matcher.py`

**What it does:**
- Reads candidate CSV from `state["candidates_csv_path"]` into a pandas DataFrame
- Expected CSV columns: `name`, `email`, `skills`, `years_experience`, `location`, `summary`
- For each candidate, constructs a text representation, runs NLP similarity against the parsed JD using LLM scoring
- Returns top N candidates ranked by score, with per-candidate match reasons

**Scoring approach — two-step:**
1. **Keyword overlap (fast pre-filter):** check `required_skills` against candidate `skills` field — compute a simple overlap ratio to filter out clearly unqualified candidates
2. **LLM scoring (depth pass):** for candidates passing the pre-filter, prompt Gemini to score the overall fit (0–100) and list top 3 match reasons

**Sample LLM scoring prompt:**
```
You are a senior technical recruiter. Score this candidate's fit for the role from 0 to 100 and give 3 specific match reasons. Return ONLY valid JSON:
{
  "score": 82,
  "match_reasons": ["...", "...", "..."],
  "flag": "strong_match" | "partial_match" | "weak_match"
}

Role requirements:
{parsed_jd}

Candidate profile:
{candidate_text}
```

**Output written to `state["ranked_candidates"]`:**
```python
[
  {"name": "...", "email": "...", "score": 87, "flag": "strong_match", "match_reasons": [...]},
  ...
]
```

**Tests to write:**
- Given a JD and 10 sample candidates, confirm output is sorted by score descending
- Confirm all output dicts contain required keys
- Test CSV with missing columns raises a clear error message

---

## Phase 4 — Agent 3: Outreach Composer

**Goal:** Write a personalised outreach email/message for each candidate in the ranked shortlist.

### File: `agents/outreach_composer.py`

**What it does:**
- Reads `state["ranked_candidates"]` and `state["parsed_jd"]`
- For each candidate (up to top 5, configurable), prompts Gemini to write a personalised outreach message
- Output: subject line + email body, referencing specific match reasons and the role

**Key implementation notes:**
- Tone should be professional but human — not a template blast
- Reference at least one specific match reason per message to make it feel personalised
- Keep emails under 200 words

**Sample prompt:**
```
Write a professional recruiter outreach email for this candidate. Reference their specific strengths. Keep it under 200 words. Return ONLY valid JSON:
{
  "subject": "...",
  "body": "..."
}

Role: {job_title} at {company}
Candidate name: {name}
Why they're a match: {match_reasons}
```

**Output written to `state["outreach_messages"]`:**
```python
[
  {"candidate_name": "...", "email": "...", "subject": "...", "body": "..."},
  ...
]
```

---

## Phase 5 — Agent 4: Interview Scheduler

**Goal:** For top shortlisted candidates, generate interview slot suggestions and structured scheduling output.

### File: `agents/interview_scheduler.py`

**What it does:**
- Reads `state["ranked_candidates"]` (top N by flag)
- Generates 3 interview slot options per candidate (next 5 business days, standard hours)
- Outputs structured data that could be passed to a calendar API or displayed in a UI

**Key implementation notes:**
- Use Python `datetime` to compute actual dates, don't hallucinate dates
- LLM is only used to generate a brief scheduling note per candidate if needed
- This agent is intentionally lightweight — its job is structured output, not reasoning

**Output written to `state["schedule"]`:**
```python
[
  {
    "candidate_name": "...",
    "slots": [
      {"date": "2026-08-10", "time": "10:00 AM", "duration_min": 45},
      {"date": "2026-08-11", "time": "2:00 PM",  "duration_min": 45},
      {"date": "2026-08-12", "time": "11:00 AM", "duration_min": 45},
    ],
    "notes": "Preferred afternoon based on timezone"
  },
  ...
]
```

---

## Phase 6 — LangGraph Pipeline (SequentialAgent)

**Goal:** Wire all four agents into a LangGraph StateGraph with sequential execution.

### File: `pipeline/graph.py`

```python
from langgraph.graph import StateGraph, END
from pipeline.state import HireFlowState
from agents.jd_parser import run_jd_parser
from agents.candidate_matcher import run_candidate_matcher
from agents.outreach_composer import run_outreach_composer
from agents.interview_scheduler import run_interview_scheduler

def build_pipeline() -> StateGraph:
    graph = StateGraph(HireFlowState)

    graph.add_node("jd_parser",           run_jd_parser)
    graph.add_node("candidate_matcher",   run_candidate_matcher)
    graph.add_node("outreach_composer",   run_outreach_composer)
    graph.add_node("interview_scheduler", run_interview_scheduler)

    graph.set_entry_point("jd_parser")
    graph.add_edge("jd_parser",           "candidate_matcher")
    graph.add_edge("candidate_matcher",   "outreach_composer")
    graph.add_edge("outreach_composer",   "interview_scheduler")
    graph.add_edge("interview_scheduler", END)

    return graph.compile()

pipeline = build_pipeline()
```

**Google ADK connection:**
- Each `run_*` function is a standard Python function wrapped as an ADK `LlmAgent` tool
- The `SequentialAgent` in ADK maps directly to this LangGraph sequential graph
- `pipeline.invoke(initial_state)` runs the full recruitment workflow end-to-end

---

## Phase 7 — FastAPI Entry Point

**Goal:** Expose the pipeline as an API so it's not just a script.

### File: `app.py`

**Endpoints:**

| Method | Route | Description |
|--------|-------|-------------|
| `GET`  | `/health` | Health check |
| `POST` | `/run` | Run the full pipeline |
| `GET`  | `/results/{run_id}` | Retrieve a previous run's output |

**`POST /run` request body:**
```json
{
  "job_description": "We are hiring a Senior ML Engineer...",
  "candidates_csv": "<base64 encoded CSV or path>"
}
```

**`POST /run` response:**
```json
{
  "run_id": "abc123",
  "ranked_candidates": [...],
  "outreach_messages": [...],
  "schedule": [...]
}
```

---

## Phase 8 — Tests

**Goal:** Every agent has at least 3 tests. Pipeline has an integration test.

```
tests/
├── test_jd_parser.py        ← unit tests for Agent 1
├── test_matcher.py          ← unit tests for Agent 2
├── test_outreach.py         ← unit tests for Agent 3
├── test_scheduler.py        ← unit tests for Agent 4
└── test_pipeline.py         ← integration: full pipeline run with sample data
```

Run with: `pytest tests/ -v`

---

## Phase 9 — README Update

The README should match the portfolio description exactly. Key sections to include:

- **What it does** — one clear paragraph matching portfolio copy
- **Architecture diagram** — ASCII or Mermaid, matching the 4-agent flow
- **Tech stack** — Google ADK, LangGraph, Gemini, FastAPI, pandas, pytest
- **How to run locally** — `.env` setup, `uvicorn app:app --reload`
- **API usage** — sample curl for `POST /run`
- **Agent breakdown** — brief description of each of the 4 agents

---

## Progress Tracker

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Repo cleanup & structure | ⬜ Not started |
| 1 | Shared state schema | ⬜ Not started |
| 2 | Agent 1 — JD Parser | ⬜ Not started |
| 3 | Agent 2 — Candidate Matcher | ⬜ Not started |
| 4 | Agent 3 — Outreach Composer | ⬜ Not started |
| 5 | Agent 4 — Interview Scheduler | ⬜ Not started |
| 6 | LangGraph pipeline wiring | ⬜ Not started |
| 7 | FastAPI entry point | ⬜ Not started |
| 8 | Tests | ⬜ Not started |
| 9 | README update | ⬜ Not started |

---

## Notes

- Use `gemini-1.5-flash` by default — fast and cheap for prototyping. Switch to `gemini-1.5-pro` for better reasoning on matcher and outreach if quality is lacking.
- All LLM calls should use structured output prompts with explicit JSON instructions. Never parse free-form text.
- The `errors` list in state is your debug trail — log it on every run.
- When the rebuild is done, update the portfolio's HireFlow GitHub link and confirm the repo README matches the card description word-for-word on the key claims (4 agents, SequentialAgent, Google ADK, LangGraph).
