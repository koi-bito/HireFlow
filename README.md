# HireFlow — Multi-Agent Recruitment System

> A production-grade, multi-agent AI pipeline that automates end-to-end recruitment workflows using **Google ADK**, **LangGraph**, and **Gemini**.

---

## What It Does

HireFlow takes a raw job description and a pool of candidate profiles and runs them through four specialised AI agents in sequence. It outputs a ranked shortlist of candidates, personalised outreach emails, and structured interview schedules — all with a single API call.

---

## Architecture

```
User Input (JD + Candidate Profiles)
        │
        ▼
┌───────────────────────┐
│   SequentialAgent     │  ← Google ADK orchestrator (LangGraph StateGraph)
└───────────┬───────────┘
            │
    ┌───────▼────────────┐
    │  Agent 1           │
    │  JD Parser         │  → extracts: role, skills, exp, location, salary
    └───────┬────────────┘
            │
    ┌───────▼────────────┐
    │  Agent 2           │
    │  Candidate Matcher │  → keyword pre-filter + LLM scoring (0–100)
    └───────┬────────────┘
            │
    ┌───────▼────────────┐
    │  Agent 3           │
    │  Outreach Composer │  → personalised email per top candidate
    └───────┬────────────┘
            │
    ┌───────▼────────────┐
    │  Agent 4           │
    │  Interview         │  → 3 real business-day slots per candidate
    │  Scheduler         │
    └────────────────────┘
            │
            ▼
    Final Output: ranked candidates + outreach drafts + schedule
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Orchestration | Google ADK + LangGraph `StateGraph` |
| LLM | Google Gemini 1.5 Flash (`google-generativeai`) |
| LLM Integration | `langchain-google-genai` |
| API Server | FastAPI + Uvicorn |
| Data Processing | pandas |
| Testing | pytest |
| Config | python-dotenv |

---

## Project Structure

```
HireFlow/
├── agents/
│   ├── jd_parser.py           ← Agent 1: JD → structured requirements
│   ├── candidate_matcher.py   ← Agent 2: keyword filter + LLM scoring
│   ├── outreach_composer.py   ← Agent 3: personalised outreach emails
│   └── interview_scheduler.py ← Agent 4: real business-day slots
├── pipeline/
│   ├── state.py               ← shared HireFlowState TypedDict
│   └── graph.py               ← LangGraph SequentialAgent wiring
├── data/
│   └── sample_candidates.csv  ← 10 sample candidates
├── tests/
│   ├── test_jd_parser.py
│   ├── test_matcher.py
│   ├── test_outreach.py
│   ├── test_scheduler.py
│   └── test_pipeline.py       ← end-to-end integration test
├── app.py                     ← FastAPI entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## Agent Breakdown

### Agent 1 — JD Parser
Receives raw job description text and prompts Gemini to extract structured fields: job title, required skills, preferred skills, years of experience, location, salary range, and employment type. Returns clean JSON, with safe defaults on failure.

### Agent 2 — Candidate Matcher
Loads candidate profiles from a CSV and runs a two-step scoring process: a fast keyword overlap pre-filter followed by an LLM scoring pass (0–100) with per-candidate match reasons and a `strong_match / partial_match / weak_match` flag.

### Agent 3 — Outreach Composer
For each top candidate (up to 5), prompts Gemini to write a personalised, under-200-word outreach email that references specific match reasons — not a generic template.

### Agent 4 — Interview Scheduler
Generates 3 concrete interview slot options per `strong_match` and `partial_match` candidate using Python `datetime` (no LLM date hallucination). Outputs calendar-ready structured data.

---

## How to Run Locally

### 1. Clone and install dependencies

```bash
git clone https://github.com/koi-bito/HireFlow.git
cd HireFlow
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

```env
GOOGLE_API_KEY=your_gemini_api_key_here
GOOGLE_GENAI_USE_VERTEXAI=FALSE
MODEL_NAME=gemini-1.5-flash
```

### 3. Start the API server

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

---

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

### Run the Full Pipeline

```bash
curl -X POST http://localhost:8000/run \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "We are hiring a Senior ML Engineer with 5+ years of Python, PyTorch, and FastAPI experience. Location: Remote. Salary: $150k–$180k. Full-time."
  }'
```

**Response:**
```json
{
  "run_id": "abc123-...",
  "parsed_jd": { "job_title": "Senior ML Engineer", "required_skills": [...], ... },
  "ranked_candidates": [
    { "name": "Alice Sharma", "score": 92, "flag": "strong_match", "match_reasons": [...] },
    ...
  ],
  "outreach_messages": [
    { "candidate_name": "Alice Sharma", "subject": "...", "body": "..." },
    ...
  ],
  "schedule": [
    { "candidate_name": "Alice Sharma", "slots": [...], "notes": "..." },
    ...
  ],
  "errors": []
}
```

### Retrieve a Previous Run

```bash
curl http://localhost:8000/results/{run_id}
```

---

## Running Tests

```bash
pytest tests/ -v
```

Tests use mocked Gemini API calls — no API key required to run the test suite.

---

## Notes

- Uses `gemini-1.5-flash` by default (fast and cost-efficient). Switch to `gemini-1.5-pro` via the `MODEL_NAME` env var for better reasoning quality on complex JDs.
- All LLM calls use structured JSON output prompts — no free-form text parsing.
- The `errors` list in each response is your debug trail — check it if results look incomplete.
- The `legacy` branch preserves the original Flask + TF-IDF prototype.
