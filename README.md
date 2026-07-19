# HireFlow - Multi-Agent Recruitment System

A multi-agent recruitment intelligence system built on Google ADK.
Paste a job description and a candidate list - five specialized agents
handle everything from JD analysis to personalized outreach drafts.

---

## Architecture

```
[Job Description + Candidates]
         │
         ▼
┌─────────────────────┐
│   JD Analyst        │  Parses JD → extracts skills, seniority, role type
│                     │  Runs initial bias check
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Bias Guardrail    │  Dedicated compliance audit
│                     │  Flags discriminatory language before pipeline continues
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Matcher           │  TF-IDF cosine similarity + LLM semantic scoring
│                     │  Combined score: 40% TF-IDF, 60% LLM
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Outreach          │  Personalized email per shortlisted candidate
│                     │  References specific skills, avoids templated language
└─────────────────────┘
         │
         ▼
[Ranked shortlist + bias report + outreach drafts]
```

---

## What Changed vs the Original

| Original | HireFlow (Rebuilt) |
|---|---|
| 9-line TF-IDF script | TF-IDF + LLM semantic scoring combined |
| Rule-based chatbot | Personalized LLM-generated outreach per candidate |
| Single Flask script | 4-agent ADK pipeline (SequentialAgent) |
| No safety checks | Dedicated bias guardrail agent |
| Static output | Structured JSON output at every stage |

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | Google ADK |
| LLM | Gemini 2.0 Flash |
| Keyword matching | scikit-learn (TF-IDF + cosine similarity) |
| Semantic scoring | Gemini via LLM agent skill |
| Language | Python 3.11 |

---

## Quickstart

Prerequisites: Python 3.11, Google AI Studio API key

```bash
git clone https://github.com/koi-bito/HireFlow.git
cd HireFlow
python -m venv hireflow_env
hireflow_env\Scripts\activate   # Windows
pip install -r requirements.txt
```

Create `.env`:
```
GOOGLE_API_KEY=your_key_here
GOOGLE_GENAI_USE_VERTEXAI=FALSE
```

Run:
```bash
python run_pipeline.py
```

---

## Project Structure

```
hireflow/
├── agents/          # Four ADK LlmAgents + SequentialAgent orchestrator
├── skills/          # Four reusable skill functions (parse, score, outreach, bias)
├── tools/           # TF-IDF matcher tool (refactored from original matcher.py)
├── data/            # Sample candidate dataset
├── tests/           # Unit tests for skills, integration tests for agents
└── run_pipeline.py  # Entry point
```

---

## Context

Started as a Flask + TF-IDF prototype. Rebuilt as a multi-agent system to
explore Google ADK's agent orchestration patterns - specifically how a
`SequentialAgent` coordinates specialized `LlmAgent` instances, and how
reusable skill functions let agents share capabilities without coupling.
The bias guardrail agent was added as a practical safety layer: discriminatory
JD language is a real problem in recruitment, and catching it before
shortlisting candidates is more useful than catching it after.
