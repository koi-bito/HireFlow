# HireFlow — Kaggle AI Agents Capstone Implementation Plan

### Goal: Extend the AI-Powered Talent Scouting & Engagement Agent into a full multi-agent recruitment system built with Google ADK, MCP servers, and agent skills — submitted to the Kaggle Vibe Coding Capstone (Agents for Business track).

### Deadline: July 6, 2026 at 11:59 PM PT | Start Date: June 20, 2026 | Total Days: 16

---

> **How to use this plan:**
> Each day has one clear goal, exact steps, and a "you're done when" checkpoint.
> There is no daily time limit — work as many hours as you need each day.
> Keep a file called `LEARNINGS.md` in your repo. Write 3–5 sentences every day — what you learned, what confused you, how you fixed it.
> **You are extending your existing repo.** Fork or branch from `koi-bito/AI-Powered-Talent-Scouting-Engagement-Agent` on Day 1.

---

## What You're Building

**HireFlow** is a multi-agent recruitment intelligence system. You are taking your existing Flask app (TF-IDF matcher + rule-based chatbot) and re-architecting it as a proper agentic pipeline using Google ADK.

The final system will have five specialized agents:

```
[Recruiter Input: Job Description]
         │
         ▼
┌─────────────────────────┐
│   JD Analyst Agent      │  Parses JD → extracts skills, seniority, role type
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Sourcing Agent        │  MCP web search → finds real candidate profiles
└────────────┬────────────┘  (replaces static CSV upload)
             │
             ▼
┌─────────────────────────┐
│   Matcher Agent         │  Your existing TF-IDF logic + LLM semantic scoring
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Outreach Agent        │  Generates personalized engagement messages
└────────────┬────────────┘  (replaces rule-based chatbot)
             │
             ▼
┌─────────────────────────┐
│   Bias Guardrail Agent  │  Flags discriminatory JD language or skewed rankings
└────────────┬────────────┘  (security/safety — required capstone concept)
             │
             ▼
     [Ranked shortlist + outreach drafts + bias report]
```

**Capstone concepts demonstrated (minimum 3 required):**

- Multi-agent system built with ADK ✓
- MCP server integration (web search) ✓
- Agent skills (reusable tool functions) ✓
- Security features (bias guardrail agent) ✓

---

## Folder Structure (Target)

```
hireflow/
├── agents/
│   ├── jd_analyst/
│   │   └── agent.py
│   ├── sourcing/
│   │   └── agent.py
│   ├── matcher/
│   │   └── agent.py
│   ├── outreach/
│   │   └── agent.py
│   ├── bias_guardrail/
│   │   └── agent.py
│   └── orchestrator/
│       └── agent.py
├── skills/
│   ├── parse_jd.py
│   ├── score_candidates.py
│   ├── draft_outreach.py
│   └── detect_bias.py
├── tools/
│   └── tfidf_matcher.py        ← Your existing matcher.py, refactored
├── data/
│   └── sample_candidates.csv
├── notebooks/
│   └── 01_adk_exploration.ipynb
├── tests/
│   ├── test_skills.py
│   └── test_agents.py
├── app.py                      ← Your existing Flask app (kept for reference)
├── chatbot.py                  ← Your existing chatbot (kept for reference)
├── matcher.py                  ← Your existing matcher (kept for reference)
├── requirements.txt
├── .env
├── .gitignore
├── LEARNINGS.md
└── README.md
```

---

## PHASE 1 — ADK Foundations + Repo Setup (Days 1–3)

> **Goal:** Understand how ADK works conceptually and get your environment running. No agent logic yet — just foundations.

---

### Day 1 — Understand ADK + Fork the Repo

**What you're learning:** How Google ADK is different from what you already know (LangChain, LangGraph), and how it structures multi-agent systems.

**Tasks:**

1. Read the ADK quickstart at `https://google.github.io/adk-docs/get-started/python/` — all of it. Don't skip anything. ADK has its own specific patterns that are different from LangGraph. Understanding the terminology now saves you hours later.

2. Key concepts to understand before continuing:
   - **LLM Agent** — the "brain." Uses Gemini to reason and call tools.
   - **Workflow Agent** — the "manager." Orchestrates other agents in sequence, parallel, or loops. Does not use an LLM itself.
   - **Custom Agent** — full Python control, inherits from `BaseAgent`.
   - **Tool** — a Python function decorated so an agent can call it.
   - **Skill** — a reusable, named capability that agents can share.
   - **Session** — how ADK tracks conversation state across turns.

3. Fork your existing repo to a new one called `hireflow-capstone` (or create a new branch called `capstone` on the same repo — either is fine).

4. Clone it locally:

```bash
git clone https://github.com/koi-bito/hireflow-capstone.git
cd hireflow-capstone
```

5. Create the full folder structure listed above. You can use:

```bash
mkdir -p agents/jd_analyst agents/sourcing agents/matcher agents/outreach agents/bias_guardrail agents/orchestrator
mkdir -p skills tools data notebooks tests
touch LEARNINGS.md
touch agents/jd_analyst/agent.py agents/sourcing/agent.py agents/matcher/agent.py
touch agents/outreach/agent.py agents/bias_guardrail/agent.py agents/orchestrator/agent.py
touch skills/parse_jd.py skills/score_candidates.py skills/draft_outreach.py skills/detect_bias.py
touch tools/tfidf_matcher.py tests/test_skills.py tests/test_agents.py
```

6. Create `LEARNINGS.md` and write your first entry: What is the difference between an LLM Agent and a Workflow Agent in ADK? When would you use each?

7. Commit everything:

```bash
git add .
git commit -m "Day 1: project structure + ADK concepts"
git push origin main
```

**You're done when:** The folder structure is live on GitHub and you can explain what a Workflow Agent vs LLM Agent is in your own words.

---

### Day 2 — Python Environment + ADK Installation

**What you're doing:** Setting up a clean virtual environment and verifying ADK is installed correctly.

**Tasks:**

1. Create and activate a virtual environment:

```bash
# Windows
py -3.11 -m venv hireflow_env
hireflow_env\Scripts\activate

# Mac/Linux
python3.11 -m venv hireflow_env
source hireflow_env/bin/activate
```

> **Why Python 3.11?** ADK currently requires Python 3.10+ and 3.11 is the most stable tested version. Do NOT use 3.13 for this project.

2. Install all dependencies:

```bash
pip install google-adk
pip install google-generativeai
pip install python-dotenv
pip install scikit-learn pandas numpy
pip install flask
pip install pytest httpx
pip install jupyter notebook ipykernel
python -m ipykernel install --user --name=hireflow_env --display-name "HireFlow"
```

3. Get a Google AI Studio API key:
   - Go to `https://aistudio.google.com/`
   - Sign in with your Google account
   - Click "Get API key" → "Create API key"
   - Copy it

4. Create a `.env` file in your project root:

```bash
GOOGLE_API_KEY=your_key_here
GOOGLE_GENAI_USE_VERTEXAI=FALSE
```

5. Create a `.gitignore`:

```
# Environment
.env
hireflow_env/
__pycache__/
*.pyc

# Data
data/raw/
*.csv

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db
```

6. Verify the full setup works:

```python
# Run this in a Python shell or quick script
import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")
response = model.generate_content("Say hello in one sentence.")
print(response.text)
# Should print something like: "Hello! How can I help you today?"
```

7. Write in `LEARNINGS.md`: What is `GOOGLE_GENAI_USE_VERTEXAI=FALSE`? Why is it set to FALSE here?

**You're done when:** The Gemini API call returns a response and your `.env` is working.

---

### Day 3 — Your First ADK Agent (Hello World)

**What you're doing:** Building the simplest possible ADK agent so you understand the pattern before you build HireFlow's agents.

**Why this day exists:** ADK has very specific conventions — how you define agents, how you run them, how tools are registered. If you skip this and jump straight to building HireFlow, you will spend hours debugging things that aren't HireFlow-specific. Learn the pattern first.

**Tasks:**

1. Create `notebooks/01_adk_exploration.ipynb` and run this:

```python
import os
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Step 1: Define a simple tool
def get_candidate_count() -> dict:
    """Returns the number of candidates in the database."""
    # Simulating a DB call
    return {"count": 42, "status": "success"}

# Step 2: Create an LLM Agent with that tool
agent = LlmAgent(
    name="recruiter_assistant",
    model="gemini-2.0-flash",
    description="A recruitment assistant that helps analyze hiring data.",
    instruction="You are a recruitment assistant. Use your tools to answer questions about candidates.",
    tools=[get_candidate_count]
)

# Step 3: Create a runner (manages the agent's execution)
session_service = InMemorySessionService()
runner = Runner(
    agent=agent,
    app_name="hireflow_test",
    session_service=session_service
)

# Step 4: Run the agent
session = session_service.create_session(app_name="hireflow_test", user_id="kunal")
user_message = types.Content(role="user", parts=[types.Part(text="How many candidates do we have?")])

print("Running agent...")
for event in runner.run(user_id="kunal", session_id=session.id, new_message=user_message):
    if event.is_final_response():
        print("Agent response:", event.response.parts[0].text)
```

2. Understand what happened:
   - The agent decided the user's question needed the `get_candidate_count` tool
   - It called the tool automatically
   - It used the tool's return value to generate a natural language response
   - You did not write any explicit tool-calling logic — ADK handled it

3. Now add a second tool and see how the agent chooses between them:

```python
def list_open_positions() -> dict:
    """Returns the list of open job positions."""
    return {
        "positions": ["Backend Engineer", "ML Engineer", "Product Manager"],
        "total": 3
    }

# Add the new tool to the agent
agent = LlmAgent(
    name="recruiter_assistant",
    model="gemini-2.0-flash",
    description="A recruitment assistant that helps analyze hiring data.",
    instruction="You are a recruitment assistant. Use your tools to answer questions about candidates and positions.",
    tools=[get_candidate_count, list_open_positions]
)

# Test: Ask something that needs the second tool
user_message = types.Content(role="user", parts=[types.Part(text="What positions are we hiring for?")])
# Re-run the runner with the new agent
```

4. Write in `LEARNINGS.md`: How does the agent know which tool to call? What would happen if two tools seemed equally relevant?

**You're done when:** Both tool calls work and you understand why ADK can route between them without explicit if-else logic.

---

## PHASE 2 — Skills + Core Agent Logic (Days 4–7)

> **Goal:** Build the four reusable skills and the first three agents. These are the heart of HireFlow.

---

### Day 4 — Refactor Your Existing Matcher into a Tool

**What you're doing:** Taking your existing `matcher.py` (TF-IDF cosine similarity) and wrapping it as a proper ADK-compatible tool. This is where your existing project work starts paying off.

**Your original `matcher.py`:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def match_with_jd(job_desc, candidates):
    skills_list = [' '.join(c['skills']) for c in candidates]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([job_desc] + skills_list)
    similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
    return similarities.flatten()
```

**Tasks:**

1. Create `tools/tfidf_matcher.py` — this is a refactored, ADK-ready version:

```python
"""
TF-IDF Candidate Matcher Tool
Original logic from matcher.py, refactored to be ADK-compatible.
Returns structured JSON instead of raw numpy arrays.
"""
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def match_candidates_to_jd(job_description: str, candidates_json: str) -> dict:
    """
    Scores a list of candidates against a job description using TF-IDF cosine similarity.

    Args:
        job_description: The full text of the job description.
        candidates_json: A JSON string of candidate objects. Each must have
                         'name' (str) and 'skills' (list of str) fields.
                         Example: '[{"name": "Alice", "skills": ["Python", "ML"]}]'

    Returns:
        A dict with a 'ranked_candidates' list, sorted by match_score descending.
    """
    try:
        candidates = json.loads(candidates_json)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid candidates JSON: {str(e)}", "ranked_candidates": []}

    if not candidates:
        return {"error": "Candidates list is empty.", "ranked_candidates": []}

    # Build text representation for each candidate
    skills_text = [' '.join(c.get('skills', [])) for c in candidates]

    # Fit TF-IDF on JD + all candidate skill texts
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([job_description] + skills_text)
    except ValueError as e:
        return {"error": f"Vectorizer error: {str(e)}", "ranked_candidates": []}

    # Cosine similarity between JD (index 0) and each candidate
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Build ranked output
    ranked = []
    for i, candidate in enumerate(candidates):
        ranked.append({
            "name": candidate.get('name', f'Candidate_{i}'),
            "skills": candidate.get('skills', []),
            "match_score": round(float(similarities[i]), 4),
            "rank": 0  # will be set after sorting
        })

    ranked.sort(key=lambda x: x['match_score'], reverse=True)
    for i, c in enumerate(ranked):
        c['rank'] = i + 1

    return {"ranked_candidates": ranked, "total": len(ranked)}
```

2. Write a quick test to make sure your refactored tool still works correctly:

```python
# Run this in your terminal or a notebook cell
from tools.tfidf_matcher import match_candidates_to_jd
import json

job_desc = "Looking for a Python developer with experience in machine learning and NLP."
candidates = [
    {"name": "Alice", "skills": ["Python", "machine learning", "scikit-learn", "NLP"]},
    {"name": "Bob", "skills": ["Java", "Spring Boot", "SQL", "REST APIs"]},
    {"name": "Priya", "skills": ["Python", "data analysis", "pandas", "NumPy"]},
]

result = match_candidates_to_jd(job_desc, json.dumps(candidates))
print(json.dumps(result, indent=2))
# Alice should rank #1, Bob should rank #3
```

3. Write in `LEARNINGS.md`: Why do we return `dict` instead of a numpy array? What does it mean for a function to be "ADK-compatible"?

**You're done when:** The refactored matcher is in `tools/tfidf_matcher.py`, tests pass, and Alice ranks #1 in your test.

---

### Day 5 — Build the Four Agent Skills

**What you're doing:** Writing the four skill functions that your agents will use as tools. Skills in ADK are just well-typed Python functions with clear docstrings. The docstring is what the LLM agent reads to decide whether to call the tool — so write them like you're explaining the tool to a smart person, not a programmer.

**Tasks:**

1. Create `skills/parse_jd.py`:

````python
"""
Skill: Parse Job Description
Extracts structured information from raw JD text.
"""
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def parse_job_description(jd_text: str) -> dict:
    """
    Parses a raw job description and extracts structured recruitment data.

    Extracts the role title, required skills, preferred skills, experience level,
    and role type (technical/non-technical). Use this when you receive a job
    description and need to understand what kind of candidate to look for.

    Args:
        jd_text: The full text of the job description.

    Returns:
        A dict with keys: role_title, required_skills (list), preferred_skills (list),
        experience_level (junior/mid/senior), role_type (technical/non-technical),
        and a brief summary.
    """
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""Extract structured information from this job description.
Respond ONLY with valid JSON — no markdown, no backticks, no explanation.

Job Description:
{jd_text}

Return this exact JSON structure:
{{
  "role_title": "string",
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["skill1", "skill2"],
  "experience_level": "junior|mid|senior",
  "role_type": "technical|non-technical",
  "summary": "one sentence summary of the role"
}}"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        return {
            "error": str(e),
            "role_title": "Unknown",
            "required_skills": [],
            "preferred_skills": [],
            "experience_level": "unknown",
            "role_type": "unknown",
            "summary": "Failed to parse JD"
        }
````

2. Create `skills/score_candidates.py`:

````python
"""
Skill: Score and Rank Candidates
Combines TF-IDF matching with LLM-based semantic scoring.
"""
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv
from tools.tfidf_matcher import match_candidates_to_jd

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def score_and_rank_candidates(job_description: str, candidates_json: str) -> dict:
    """
    Scores candidates against a job description using two methods:
    TF-IDF cosine similarity (fast keyword matching) and LLM semantic scoring
    (deep understanding of fit). Returns a combined ranked shortlist.

    Use this after you have a list of candidates and a job description,
    when you need to identify the best-fit candidates to prioritize.

    Args:
        job_description: The full text of the job description.
        candidates_json: JSON string of candidate list. Each candidate needs
                         'name' (str) and 'skills' (list of str).

    Returns:
        A dict with 'shortlist' — top candidates with tfidf_score, llm_score,
        combined_score, and a brief fit_reason from the LLM.
    """
    # Step 1: TF-IDF matching (your existing logic)
    tfidf_result = match_candidates_to_jd(job_description, candidates_json)
    if "error" in tfidf_result:
        return tfidf_result

    ranked = tfidf_result["ranked_candidates"]
    # Take top 5 for LLM re-scoring
    top_candidates = ranked[:5]

    # Step 2: LLM semantic scoring on the top 5
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""You are a senior recruiter. Score each candidate's fit for this job.

Job Description:
{job_description}

Candidates (top 5 by keyword match):
{json.dumps(top_candidates, indent=2)}

For each candidate, give:
- llm_score: float between 0.0 and 1.0 (1.0 = perfect fit)
- fit_reason: one sentence explaining the score

Respond ONLY with valid JSON. No markdown, no backticks.

Return this structure:
{{
  "scored": [
    {{"name": "...", "llm_score": 0.0, "fit_reason": "..."}}
  ]
}}"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        llm_scores = json.loads(raw.strip())

        # Merge TF-IDF and LLM scores
        llm_map = {s["name"]: s for s in llm_scores.get("scored", [])}
        shortlist = []
        for c in top_candidates:
            llm_data = llm_map.get(c["name"], {"llm_score": 0.5, "fit_reason": "Not scored by LLM"})
            combined = round(0.4 * c["match_score"] + 0.6 * llm_data["llm_score"], 4)
            shortlist.append({
                "name": c["name"],
                "skills": c["skills"],
                "tfidf_score": c["match_score"],
                "llm_score": llm_data["llm_score"],
                "combined_score": combined,
                "fit_reason": llm_data["fit_reason"],
                "rank": 0
            })

        shortlist.sort(key=lambda x: x["combined_score"], reverse=True)
        for i, c in enumerate(shortlist):
            c["rank"] = i + 1

        return {"shortlist": shortlist, "total": len(shortlist)}

    except Exception as e:
        return {"error": str(e), "shortlist": top_candidates}
````

3. Create `skills/draft_outreach.py`:

````python
"""
Skill: Draft Candidate Outreach
Generates personalized outreach messages for shortlisted candidates.
"""
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def draft_outreach_message(candidate_name: str, candidate_skills_json: str,
                           role_title: str, company_name: str,
                           fit_reason: str) -> dict:
    """
    Generates a personalized, professional outreach message for a candidate.
    The message highlights the candidate's specific relevant skills and explains
    why they are a strong match for the role.

    Use this for each shortlisted candidate after scoring is complete,
    to generate the initial recruiter outreach message.

    Args:
        candidate_name: Full name of the candidate.
        candidate_skills_json: JSON string of the candidate's skills list.
        role_title: The title of the open position.
        company_name: The name of the hiring company.
        fit_reason: A brief sentence on why this candidate fits (from scorer).

    Returns:
        A dict with 'subject' (email subject line) and 'message' (email body).
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    skills = json.loads(candidate_skills_json)

    prompt = f"""Write a personalized recruiter outreach email for this candidate.

Candidate Name: {candidate_name}
Candidate Skills: {', '.join(skills)}
Role: {role_title}
Company: {company_name}
Why They Fit: {fit_reason}

Rules:
- Keep it under 150 words
- Reference 2-3 of their specific skills by name
- Sound human and conversational, not corporate
- Do NOT use phrases like "I came across your profile" or "exciting opportunity"
- End with a soft call to action (ask if they're open to a chat, not a formal interview)

Respond ONLY with valid JSON. No markdown, no backticks.

Return:
{{
  "subject": "email subject line here",
  "message": "full email body here"
}}"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        return {
            "subject": f"{role_title} at {company_name} — Would love to connect",
            "message": f"Hi {candidate_name}, we think your background in {skills[0] if skills else 'your field'} could be a great fit for our {role_title} role. Would you be open to a quick chat?",
            "error": str(e)
        }
````

4. Create `skills/detect_bias.py`:

````python
"""
Skill: Bias Detection Guardrail
Flags discriminatory language in JDs and checks for skewed candidate rankings.
This is the security/safety feature required by the capstone.
"""
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Hardcoded patterns that are immediately flagged without LLM call
# (faster and more reliable for known patterns)
KNOWN_BIAS_PATTERNS = [
    "young", "energetic team", "digital native", "recent graduate only",
    "native english", "he or she", "brotherhood", "gentleman", "guys",
    "rockstar", "ninja", "culture fit", "must be local"
]


def detect_bias_in_jd(jd_text: str) -> dict:
    """
    Analyzes a job description for potentially discriminatory or biased language.
    Flags both hardcoded known-bias patterns and uses an LLM to detect
    subtler forms of exclusionary language. This is a safety guardrail —
    always run this before publishing a JD or sending candidate shortlists.

    Args:
        jd_text: The full text of the job description.

    Returns:
        A dict with 'bias_detected' (bool), 'severity' (low/medium/high),
        'flags' (list of specific issues found), and 'suggestions' (list of fixes).
    """
    flags = []

    # Pattern check (fast, no LLM needed)
    jd_lower = jd_text.lower()
    for pattern in KNOWN_BIAS_PATTERNS:
        if pattern in jd_lower:
            flags.append(f"Known bias pattern detected: '{pattern}'")

    # LLM check (catches subtler issues)
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""You are an HR compliance expert. Analyze this job description for bias.

Job Description:
{jd_text}

Check for:
1. Age bias (explicitly or implicitly targeting young or old candidates)
2. Gender bias (gendered language, gendered role assumptions)
3. Nationality/origin bias
4. Disability exclusion
5. Culture-fit language that excludes minorities
6. Unnecessarily restrictive requirements that aren't actually needed for the role

Respond ONLY with valid JSON. No markdown, no backticks.

Return:
{{
  "additional_flags": ["specific issue 1", "specific issue 2"],
  "severity": "none|low|medium|high",
  "suggestions": ["suggestion 1", "suggestion 2"],
  "overall_assessment": "one sentence"
}}"""

    try:
        response = model.generate_content(prompt)
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        llm_result = json.loads(raw.strip())

        all_flags = flags + llm_result.get("additional_flags", [])
        severity = llm_result.get("severity", "low") if all_flags else "none"

        return {
            "bias_detected": len(all_flags) > 0,
            "severity": severity,
            "flags": all_flags,
            "suggestions": llm_result.get("suggestions", []),
            "overall_assessment": llm_result.get("overall_assessment", "No issues found.")
        }

    except Exception as e:
        return {
            "bias_detected": len(flags) > 0,
            "severity": "unknown",
            "flags": flags,
            "suggestions": [],
            "overall_assessment": f"LLM check failed: {str(e)}. Only pattern check was run.",
            "error": str(e)
        }
````

5. Test all four skills end-to-end in a notebook. Use a real job description for testing (copy any SDE JD from LinkedIn).

**You're done when:** All four skills return sensible output when tested with a real JD. Commit everything.

---

### Day 6 — Build the JD Analyst + Bias Guardrail Agents

**What you're doing:** Building the first two ADK agents. These are the simplest ones — each wraps a single skill. Understanding this pattern makes Days 7 and 8 easier.

**Tasks:**

1. Create `agents/jd_analyst/agent.py`:

```python
"""
JD Analyst Agent
Parses a raw job description into structured recruitment data.
"""
from google.adk.agents import LlmAgent
from skills.parse_jd import parse_job_description
from skills.detect_bias import detect_bias_in_jd

jd_analyst_agent = LlmAgent(
    name="jd_analyst_agent",
    model="gemini-2.0-flash",
    description=(
        "Analyzes job descriptions. Extracts structured data (required skills, "
        "seniority, role type) and runs a bias check. Use this first, before "
        "any candidate sourcing or matching."
    ),
    instruction="""You are a senior HR analyst specializing in job description analysis.

When given a job description:
1. Call parse_job_description to extract structured requirements
2. Call detect_bias_in_jd to check for discriminatory language
3. If bias is detected at medium or high severity, highlight it clearly in your response
4. Summarize the key requirements for the recruiter in a clear, brief format

Be direct. Don't pad your response. Recruiters are busy.""",
    tools=[parse_job_description, detect_bias_in_jd]
)
```

2. Create `agents/bias_guardrail/agent.py`:

```python
"""
Bias Guardrail Agent
A dedicated safety agent that audits JDs and shortlists for bias.
Runs independently from the main pipeline as a final check.
"""
from google.adk.agents import LlmAgent
from skills.detect_bias import detect_bias_in_jd

bias_guardrail_agent = LlmAgent(
    name="bias_guardrail_agent",
    model="gemini-2.0-flash",
    description=(
        "A compliance and safety agent that checks job descriptions and "
        "candidate shortlists for bias, discrimination, and unfair exclusion. "
        "Always run this before finalizing any recruitment output."
    ),
    instruction="""You are an HR compliance officer focused on fair hiring practices.

Your job is to catch anything that could expose the company to discrimination claims
or that unfairly excludes qualified candidates.

When given a JD:
- Run detect_bias_in_jd
- If severity is 'none' or 'low': confirm the JD is clean with brief notes
- If severity is 'medium': list each flag and give specific rewrite suggestions
- If severity is 'high': clearly warn that this JD should NOT be published as-is

Be specific. Generic advice is useless. Point to exact phrases that are problematic.""",
    tools=[detect_bias_in_jd]
)
```

3. Test both agents in isolation using the ADK runner:

```python
# Test in a notebook
import os
from dotenv import load_dotenv
load_dotenv()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from agents.jd_analyst.agent import jd_analyst_agent

session_service = InMemorySessionService()
runner = Runner(
    agent=jd_analyst_agent,
    app_name="hireflow",
    session_service=session_service
)

session = session_service.create_session(app_name="hireflow", user_id="test")

jd = """
We are looking for a young, energetic Backend Engineer to join our brotherhood.
You must be a rockstar Python developer with 5+ years experience.
Native English speaker preferred.
"""

message = types.Content(role="user", parts=[types.Part(text=f"Analyze this JD: {jd}")])

for event in runner.run(user_id="test", session_id=session.id, new_message=message):
    if event.is_final_response():
        print(event.response.parts[0].text)
```

> Note: The test JD above is intentionally full of bias markers ("young", "brotherhood", "rockstar", "Native English speaker preferred"). Your agent should catch all of them.

4. Write in `LEARNINGS.md`: The JD Analyst has two tools. How does the agent decide the order to call them? Could the order matter?

**You're done when:** Both agents run and the bias guardrail catches all the red flags in the test JD.

---

### Day 7 — Build the Matcher + Outreach Agents

**What you're doing:** Building the two remaining specialized agents. The Matcher agent is more complex because it needs to handle candidate data.

**Tasks:**

1. Create `agents/matcher/agent.py`:

```python
"""
Matcher Agent
Scores and ranks candidates against a job description.
Uses TF-IDF + LLM semantic scoring in combination.
"""
from google.adk.agents import LlmAgent
from skills.score_candidates import score_and_rank_candidates
from tools.tfidf_matcher import match_candidates_to_jd

matcher_agent = LlmAgent(
    name="matcher_agent",
    model="gemini-2.0-flash",
    description=(
        "Scores and ranks candidates against a job description. "
        "Combines keyword-based TF-IDF matching with LLM semantic scoring "
        "to produce a ranked shortlist. Use after you have a list of candidates."
    ),
    instruction="""You are a technical recruiter who specializes in candidate evaluation.

When given a job description and a list of candidates:
1. Call score_and_rank_candidates to get the ranked shortlist
2. Present the top 3 candidates clearly: their rank, combined score, and fit reason
3. If any candidate scores below 0.3 combined, note they are a weak match
4. Always mention the scoring method used (TF-IDF + LLM semantic scoring)

Format the shortlist as a clean, scannable list. Recruiters need to act on this fast.""",
    tools=[score_and_rank_candidates, match_candidates_to_jd]
)
```

2. Create `agents/outreach/agent.py`:

```python
"""
Outreach Agent
Generates personalized candidate engagement messages.
Replaces the rule-based chatbot simulation from the original project.
"""
import json
from google.adk.agents import LlmAgent
from skills.draft_outreach import draft_outreach_message

outreach_agent = LlmAgent(
    name="outreach_agent",
    model="gemini-2.0-flash",
    description=(
        "Generates personalized outreach messages for shortlisted candidates. "
        "Produces a unique email for each candidate that references their specific "
        "skills and explains why they were chosen. Use after ranking is complete."
    ),
    instruction="""You are an expert recruitment copywriter. Your outreach messages
have high response rates because they feel human and specific, not templated.

When given a shortlist of candidates and a role:
1. Call draft_outreach_message for each candidate in the shortlist
2. Present each message with the candidate name as a header
3. If a candidate's skills list is empty, note that more profile data is needed
   before drafting outreach

Your output is the final step before a human recruiter clicks send.
Make it count.""",
    tools=[draft_outreach_message]
)
```

3. Create `data/sample_candidates.csv` with test data:

```csv
name,skills
Alice Sharma,"Python,machine learning,scikit-learn,NLP,transformers,FastAPI"
Bob Zhang,"Java,Spring Boot,SQL,REST APIs,Docker,Kubernetes"
Priya Mehta,"Python,data analysis,pandas,NumPy,SQL,Tableau"
Carlos Rivera,"JavaScript,React,Node.js,TypeScript,GraphQL,CSS"
Fatima Al-Hassan,"Python,deep learning,PyTorch,computer vision,OpenCV"
Ravi Kumar,"DevOps,AWS,Terraform,CI/CD,GitHub Actions,Linux"
Mei Lin,"Product Management,Agile,Jira,user research,roadmapping"
Arjun Patel,"Python,Flask,Django,PostgreSQL,Redis,Docker"
Sarah O'Brien,"Machine Learning,XGBoost,feature engineering,A/B testing,Python"
Dmitri Volkov,"C++,embedded systems,RTOS,firmware,IoT"
```

4. Write a test that loads the CSV and runs the full matcher agent against a real JD:

```python
import pandas as pd
import json

# Load candidates
df = pd.read_csv("data/sample_candidates.csv")
candidates = []
for _, row in df.iterrows():
    candidates.append({
        "name": row["name"],
        "skills": [s.strip() for s in row["skills"].split(",")]
    })

jd = """
We are looking for a Senior ML Engineer with strong Python skills.
The ideal candidate has experience with machine learning frameworks,
model training, feature engineering, and deploying models to production.
Experience with NLP or computer vision is a plus.
"""

# Run matcher agent with these candidates
candidates_json = json.dumps(candidates)
message_text = f"Rank these candidates for this JD:\n\nJD:\n{jd}\n\nCandidates:\n{candidates_json}"
```

5. Write in `LEARNINGS.md`: The Outreach Agent needs to call `draft_outreach_message` once per candidate. How would you handle this if the shortlist had 10 candidates? What ADK feature might help?

**You're done when:** All four specialized agents are built and individually tested. Commit everything.

---

## PHASE 3 — Orchestrator + Multi-Agent Pipeline (Days 8–10)

> **Goal:** Wire all agents together into the HireFlow pipeline using an ADK SequentialAgent orchestrator.

---

### Day 8 — Build the Orchestrator Agent

**What you're doing:** Creating the "manager" agent that coordinates the other four agents in sequence. This is what makes HireFlow a true multi-agent system — one input, automatic hand-off between agents, one final output.

**Key concept:** The orchestrator is a `SequentialAgent` — a Workflow Agent, not an LLM Agent. It does not think or make decisions. It just runs agents in order and passes state between them. The intelligence is in the individual agents.

**Tasks:**

1. Create `agents/orchestrator/agent.py`:

```python
"""
HireFlow Orchestrator
Coordinates the full recruitment pipeline:
JD Analysis → Bias Check → Candidate Matching → Outreach Drafting

Uses SequentialAgent (Workflow Agent) to ensure correct execution order.
"""
from google.adk.agents import SequentialAgent
from agents.jd_analyst.agent import jd_analyst_agent
from agents.bias_guardrail.agent import bias_guardrail_agent
from agents.matcher.agent import matcher_agent
from agents.outreach.agent import outreach_agent

hireflow_pipeline = SequentialAgent(
    name="hireflow_pipeline",
    description=(
        "The HireFlow multi-agent recruitment pipeline. "
        "Given a job description and candidate list, runs JD analysis, "
        "bias checking, candidate matching, and outreach drafting in sequence. "
        "Returns a complete recruitment package: parsed JD, bias report, "
        "ranked shortlist, and personalized outreach messages."
    ),
    sub_agents=[
        jd_analyst_agent,       # Step 1: Parse JD + check for bias
        bias_guardrail_agent,   # Step 2: Dedicated bias audit (double check)
        matcher_agent,          # Step 3: Score and rank candidates
        outreach_agent          # Step 4: Draft outreach for top candidates
    ]
)
```

2. Test the full pipeline end-to-end:

```python
import os, json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from agents.orchestrator.agent import hireflow_pipeline

# Load candidates
df = pd.read_csv("data/sample_candidates.csv")
candidates = [
    {"name": row["name"], "skills": [s.strip() for s in row["skills"].split(",")]}
    for _, row in df.iterrows()
]

jd = """Senior Python Developer — We need someone with strong Python,
machine learning, and API development experience to join our team.
2+ years of professional experience required. Experience with NLP is a plus."""

# Build the input message
input_text = f"""Process this recruitment request:

JOB DESCRIPTION:
{jd}

CANDIDATES:
{json.dumps(candidates, indent=2)}

Run the full pipeline: analyze the JD, check for bias, rank candidates, and draft outreach."""

session_service = InMemorySessionService()
runner = Runner(
    agent=hireflow_pipeline,
    app_name="hireflow",
    session_service=session_service
)
session = session_service.create_session(app_name="hireflow", user_id="recruiter")
message = types.Content(role="user", parts=[types.Part(text=input_text)])

print("Running HireFlow pipeline...\n")
for event in runner.run(user_id="recruiter", session_id=session.id, new_message=message):
    if event.is_final_response():
        print("=== HIREFLOW OUTPUT ===")
        print(event.response.parts[0].text)
```

3. When it runs, verify the output contains all four stages: JD analysis, bias report, ranked shortlist, and outreach messages.

4. Write in `LEARNINGS.md`: What would happen if you used a `ParallelAgent` instead of `SequentialAgent` here? Which steps could theoretically run in parallel? Which ones can't?

**You're done when:** The full pipeline runs end-to-end and produces output from all four stages.

---

### Day 9 — Add MCP Web Search for Candidate Sourcing

**What you're doing:** Adding real candidate sourcing via MCP web search. This replaces the static CSV upload from your original project with a live sourcing agent that searches the web. This is one of the core capstone requirements.

**What MCP is:** Model Context Protocol is a standard for connecting LLM agents to external tools and services. Google ADK has native MCP support. By using MCP web search, your agent can search LinkedIn profiles, GitHub, and public directories — rather than being limited to a CSV you upload manually.

**Tasks:**

1. First, understand how ADK integrates with MCP tools. Read: `https://google.github.io/adk-docs/tools/mcp-tools/`

2. Update `agents/sourcing/agent.py` to use MCP web search:

```python
"""
Sourcing Agent
Searches the web for candidate profiles matching a given job description.
Uses MCP Google Search to find real candidates from public sources.
This replaces the static CSV upload from the original project.
"""
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters

# MCP server for Google Search
# This uses the publicly available Google Search MCP server
search_toolset = MCPToolset(
    connection_params=StdioServerParameters(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-google-search"],
        env={
            "GOOGLE_API_KEY": __import__('os').getenv("GOOGLE_API_KEY"),
            "GOOGLE_CSE_ID": __import__('os').getenv("GOOGLE_CSE_ID")
        }
    )
)

sourcing_agent = LlmAgent(
    name="sourcing_agent",
    model="gemini-2.0-flash",
    description=(
        "Searches the web for candidate profiles that match a job description. "
        "Uses Google Search via MCP to find LinkedIn profiles, GitHub profiles, "
        "and other public candidate data. Use this when you need to source "
        "candidates without a pre-existing candidate database."
    ),
    instruction="""You are a technical sourcer at a top recruitment firm.

When given a job description:
1. Identify the 3-4 most important skills from the JD
2. Search for candidates using queries like:
   - "Python ML engineer GitHub portfolio site:github.com"
   - "LinkedIn profile Python machine learning engineer"
3. From search results, extract candidate information:
   - Name (if visible)
   - Skills/technologies mentioned
   - Link to their profile
4. Return a structured list of up to 10 sourced candidates

If search results are limited, note what additional sourcing channels a human recruiter could use.
Be honest about what you found vs what you inferred.""",
    tools=[search_toolset]
)
```

3. Add the required environment variable to your `.env`:

```bash
GOOGLE_API_KEY=your_key_here
GOOGLE_GENAI_USE_VERTEXAI=FALSE
GOOGLE_CSE_ID=your_custom_search_engine_id_here
```

> **To get a Google CSE ID:** Go to `https://programmablesearchengine.google.com/`, create a search engine, and copy the ID. Set it to search the entire web.

4. Install the MCP server dependency:

```bash
# Make sure Node.js is installed (needed for npx)
node --version   # Should show v18+
npm install -g @modelcontextprotocol/server-google-search
```

> **If Node.js is not installed:** Download from `https://nodejs.org/` — LTS version. This is required for MCP stdio servers.

5. Test the sourcing agent:

```python
# In a notebook
from agents.sourcing.agent import sourcing_agent

jd = "Senior Python developer with machine learning and NLP experience."
message_text = f"Find candidates for this role: {jd}"
# Run with runner as before
```

6. Write in `LEARNINGS.md`: What is MCP? Why is it useful for agents compared to writing a custom web search function? What's the difference between MCP tools and regular Python tools in ADK?

**You're done when:** The sourcing agent can run a web search and return a structured list of candidate profiles. Commit.

---

### Day 10 — Update Orchestrator + Wire Sourcing In + Full E2E Test

**What you're doing:** Adding the sourcing agent to the orchestrator and testing the complete 5-agent pipeline. This is the core integration milestone.

**Tasks:**

1. Update `agents/orchestrator/agent.py` to include the sourcing agent:

```python
from google.adk.agents import SequentialAgent
from agents.jd_analyst.agent import jd_analyst_agent
from agents.bias_guardrail.agent import bias_guardrail_agent
from agents.sourcing.agent import sourcing_agent
from agents.matcher.agent import matcher_agent
from agents.outreach.agent import outreach_agent

hireflow_pipeline = SequentialAgent(
    name="hireflow_pipeline",
    description=(
        "The HireFlow multi-agent recruitment pipeline. Given a job description, "
        "runs: JD analysis → bias check → candidate sourcing → candidate matching "
        "→ outreach drafting. Returns a complete recruitment package."
    ),
    sub_agents=[
        jd_analyst_agent,       # Step 1: Parse JD + initial bias check
        bias_guardrail_agent,   # Step 2: Dedicated compliance audit
        sourcing_agent,         # Step 3: Search for candidates via MCP
        matcher_agent,          # Step 4: Score and rank candidates
        outreach_agent          # Step 5: Draft outreach messages
    ]
)
```

2. Run the complete 5-agent pipeline with only a JD as input (no pre-loaded CSV):

```python
jd = """
Senior Machine Learning Engineer

We are looking for an ML Engineer to join our AI team.
Required: Python, PyTorch or TensorFlow, experience with model training and deployment.
Nice to have: NLP, computer vision, MLflow, Docker.
3+ years of professional ML experience.
"""

input_text = f"""Run the full HireFlow pipeline for this job description.
Source candidates from the web, then rank and draft outreach.

JOB DESCRIPTION:
{jd}"""

# Run with orchestrator as before
```

3. The output should now flow through all five agents:
   - JD Analyst: parses the JD, runs initial bias check
   - Bias Guardrail: confirms no compliance issues
   - Sourcing Agent: searches for candidates via MCP
   - Matcher Agent: scores the sourced candidates
   - Outreach Agent: drafts messages for the top 3

4. Document the full output in your `LEARNINGS.md`. What worked well? What did the sourcing agent struggle with?

5. Commit with a clear message:

```bash
git add .
git commit -m "Day 10: complete 5-agent pipeline working end-to-end"
git push origin main
```

**You're done when:** The 5-agent pipeline runs from a single JD input and produces output from every stage.

---

## PHASE 4 — Tests + Kaggle Notebook + Polish (Days 11–14)

> **Goal:** Write tests, package the project as a Kaggle notebook, and make everything submission-ready.

---

### Day 11 — Write Tests for Skills and Agents

**What you're doing:** Writing tests for your skills and agents. The capstone submission is judged partly on engineering quality. Tests show that you built this seriously, not just hacked it together.

**Tasks:**

1. Create `tests/test_skills.py`:

```python
"""
Tests for HireFlow skills.
Run with: pytest tests/test_skills.py -v
"""
import json
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.tfidf_matcher import match_candidates_to_jd


class TestTFIDFMatcher:
    """Tests for the refactored TF-IDF matcher tool."""

    def test_basic_ranking(self):
        """Alice (Python ML) should rank higher than Bob (Java) for a Python ML JD."""
        jd = "Python developer with machine learning and NLP experience required."
        candidates = [
            {"name": "Alice", "skills": ["Python", "machine learning", "NLP", "scikit-learn"]},
            {"name": "Bob", "skills": ["Java", "Spring Boot", "SQL"]}
        ]
        result = match_candidates_to_jd(jd, json.dumps(candidates))

        assert "ranked_candidates" in result
        assert result["ranked_candidates"][0]["name"] == "Alice", \
            "Alice should rank #1 for a Python ML role"

    def test_empty_candidates(self):
        """Should handle empty candidate list gracefully."""
        result = match_candidates_to_jd("Python developer", json.dumps([]))
        assert "error" in result

    def test_invalid_json(self):
        """Should handle malformed JSON gracefully."""
        result = match_candidates_to_jd("Python developer", "not valid json")
        assert "error" in result

    def test_returns_all_candidates(self):
        """Result should contain all input candidates."""
        jd = "Python developer"
        candidates = [
            {"name": "A", "skills": ["Python"]},
            {"name": "B", "skills": ["Java"]},
            {"name": "C", "skills": ["Ruby"]}
        ]
        result = match_candidates_to_jd(jd, json.dumps(candidates))
        assert result["total"] == 3

    def test_scores_between_zero_and_one(self):
        """All match scores should be between 0 and 1."""
        jd = "Python developer"
        candidates = [{"name": "Alice", "skills": ["Python", "Django"]}]
        result = match_candidates_to_jd(jd, json.dumps(candidates))
        for c in result["ranked_candidates"]:
            assert 0.0 <= c["match_score"] <= 1.0


class TestBiasDetection:
    """Tests for the bias detection skill."""

    def test_known_bias_pattern_detected(self):
        """Should catch 'young' as a known bias pattern."""
        from skills.detect_bias import detect_bias_in_jd
        jd = "We are looking for a young, energetic developer to join our team."
        result = detect_bias_in_jd(jd)
        assert result["bias_detected"] is True
        assert any("young" in flag.lower() for flag in result["flags"])

    def test_clean_jd_passes(self):
        """A clean JD should not trigger bias flags."""
        from skills.detect_bias import detect_bias_in_jd
        jd = """Senior Software Engineer. Required: 3+ years Python experience,
        strong problem-solving skills, experience with REST APIs and SQL databases."""
        result = detect_bias_in_jd(jd)
        # Clean JD should have low or no severity
        assert result["severity"] in ["none", "low"]
```

2. Create `tests/test_agents.py`:

```python
"""
Integration tests for HireFlow agents.
These tests call the actual ADK agents — they require a valid GOOGLE_API_KEY.
Run with: pytest tests/test_agents.py -v
"""
import json
import pytest
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


def run_agent(agent, message_text: str) -> str:
    """Helper: runs an agent and returns its final response text."""
    session_service = InMemorySessionService()
    runner = Runner(agent=agent, app_name="test", session_service=session_service)
    session = session_service.create_session(app_name="test", user_id="test_user")
    message = types.Content(role="user", parts=[types.Part(text=message_text)])

    for event in runner.run(user_id="test_user", session_id=session.id, new_message=message):
        if event.is_final_response():
            return event.response.parts[0].text
    return ""


class TestJDAnalystAgent:
    """Tests for the JD Analyst Agent."""

    def test_extracts_role_title(self):
        """Agent should mention the role title in its response."""
        from agents.jd_analyst.agent import jd_analyst_agent
        jd = "We are hiring a Senior Python Developer with 5+ years of experience."
        response = run_agent(jd_analyst_agent, f"Analyze this JD: {jd}")
        assert "python" in response.lower() or "developer" in response.lower()

    def test_flags_biased_jd(self):
        """Agent should flag known bias patterns."""
        from agents.jd_analyst.agent import jd_analyst_agent
        jd = "Looking for a young rockstar developer. Must be a native English speaker."
        response = run_agent(jd_analyst_agent, f"Analyze this JD: {jd}")
        # Response should contain bias-related words
        bias_words = ["bias", "discriminat", "young", "native", "rockstar", "flag"]
        assert any(word in response.lower() for word in bias_words)


class TestMatcherAgent:
    """Tests for the Matcher Agent."""

    def test_returns_ranking(self):
        """Matcher should return a ranked list."""
        from agents.matcher.agent import matcher_agent
        jd = "Python machine learning engineer with scikit-learn experience."
        candidates = [
            {"name": "Alice", "skills": ["Python", "machine learning", "scikit-learn"]},
            {"name": "Bob", "skills": ["Java", "Spring"]}
        ]
        msg = f"Rank these candidates:\nJD: {jd}\nCandidates: {json.dumps(candidates)}"
        response = run_agent(matcher_agent, msg)
        assert "alice" in response.lower()
        assert "bob" in response.lower()
```

3. Run the tests:

```bash
# Skill tests (fast, no API calls needed)
pytest tests/test_skills.py -v

# Agent tests (slower, needs GOOGLE_API_KEY)
pytest tests/test_agents.py -v
```

4. All skill tests must pass. Agent tests may occasionally fail due to LLM non-determinism — that's acceptable, but they should pass most of the time.

5. Write in `LEARNINGS.md`: Why do we test skills separately from agents? What makes agent tests harder to write than function tests?

**You're done when:** All skill tests pass. At least 2 of the agent tests pass. Commit.

---

### Day 12 — Build the Kaggle Notebook

**What you're doing:** Creating the actual submission artifact — a Kaggle notebook that demonstrates the full pipeline. The Kaggle submission is a notebook, not a repo. This day is critical.

**Tasks:**

1. Go to `kaggle.com`, open the competition, and create a new notebook.

2. Structure your Kaggle notebook exactly like this (each section is a markdown cell followed by code):

```
Section 1: Project Overview
- What HireFlow is
- The problem it solves
- Architecture diagram (draw it in ASCII or paste the one from this plan)

Section 2: Setup + Installation
- pip installs
- Environment setup (paste API key setup without the key itself)

Section 3: Concept 1 — Multi-Agent System with ADK
- Paste and run your jd_analyst_agent
- Show it analyzing a sample JD
- Explain what ADK SequentialAgent and LlmAgent are

Section 4: Concept 2 — MCP Server Integration
- Paste and run your sourcing_agent
- Show a real MCP web search being executed
- Explain what MCP is and why it matters for agents

Section 5: Concept 3 — Agent Skills
- Paste and run your four skills individually
- Show parse_jd, score_candidates, draft_outreach, detect_bias in action
- Explain the skills pattern in ADK

Section 6: Concept 4 — Security (Bias Guardrail)
- Run the bias_guardrail_agent on a biased JD
- Show it catching specific flags
- Explain why this is a safety/security feature

Section 7: Full Pipeline Demo
- Run the complete hireflow_pipeline on a realistic use case
- Show all 5 agent outputs
- Note the before vs after: original project vs HireFlow

Section 8: Conclusion
- What you built
- What you'd add next (LinkedIn API, real ATS integration, async pipeline)
- Link to your GitHub repo
```

3. For the Kaggle notebook, since you can't use a `.env` file, use Kaggle Secrets instead:
   - In your notebook, click the padlock icon → "Add a new secret"
   - Add `GOOGLE_API_KEY` as a secret
   - In code, access it with:

```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
GOOGLE_API_KEY = secrets.get_secret("GOOGLE_API_KEY")

import os
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
```

4. Make sure the notebook runs fully from top to bottom without errors before you submit.

**You're done when:** The Kaggle notebook runs clean, end-to-end, with output visible for every section.

---

### Day 13 — GitHub README + Architecture Diagram

**What you're doing:** Making the GitHub repo submission-ready. Judges and community members will look at the repo. A strong README is the difference between looking like a hobbyist and looking like an engineer.

**Tasks:**

1. Rewrite your `README.md` completely. Use this structure:

```markdown
# HireFlow — Multi-Agent Recruitment Intelligence System

> Kaggle AI Agents Capstone | Agents for Business Track

An intelligent, multi-agent recruitment system that automates the full pipeline from job description analysis to personalized candidate outreach. Built on Google ADK with MCP web search integration.

## Problem

Traditional ATS tools are keyword sieves — they rank by exact match and miss strong candidates with adjacent skills. And drafting outreach for 50 candidates manually takes hours.

## Solution

HireFlow uses five specialized agents that collaborate:

| Agent          | Role                                | ADK Type       |
| -------------- | ----------------------------------- | -------------- |
| JD Analyst     | Parses JD, extracts requirements    | LlmAgent       |
| Bias Guardrail | Audits for discriminatory language  | LlmAgent       |
| Sourcing Agent | Searches web for candidates via MCP | LlmAgent + MCP |
| Matcher Agent  | Scores candidates (TF-IDF + LLM)    | LlmAgent       |
| Outreach Agent | Drafts personalized messages        | LlmAgent       |

## Capstone Concepts Demonstrated

- **Multi-agent system (ADK):** SequentialAgent orchestrating 5 LlmAgents
- **MCP server:** Google Search MCP for live candidate sourcing
- **Agent skills:** 4 reusable skills shared across agents
- **Security feature:** Dedicated bias guardrail agent with pattern detection + LLM audit

## Architecture

[Paste your architecture diagram here]

## Quickstart

[Setup instructions]

## Before vs After

| Original Project   | HireFlow                            |
| ------------------ | ----------------------------------- |
| Static CSV upload  | Live web sourcing via MCP           |
| TF-IDF only        | TF-IDF + LLM semantic scoring       |
| Rule-based chatbot | Personalized LLM-generated outreach |
| Single script      | 5-agent ADK pipeline                |
| No safety checks   | Bias guardrail agent                |

## Tech Stack

- Google ADK (multi-agent orchestration)
- Gemini 2.0 Flash (LLM)
- MCP Google Search (candidate sourcing)
- scikit-learn (TF-IDF matching)
- Python 3.11

## Kaggle Submission

[Link to Kaggle notebook]
```

2. Commit the README:

```bash
git add README.md
git commit -m "Day 13: complete README for submission"
git push origin main
```

**You're done when:** The README tells a clear story and includes the before/after comparison table.

---

### Day 14 — Fix Bugs + Final End-to-End Test

**What you're doing:** One full day dedicated to fixing anything that broke, cleaning up code, and running a final complete test of everything.

**Tasks:**

1. Run the full pipeline once more from scratch:
   - Fresh Python session
   - Only a JD as input
   - Pipeline runs through all 5 agents
   - Output is coherent and complete

2. Checklist — go through every item:
   - [ ] `tests/test_skills.py` — all tests pass
   - [ ] `tests/test_agents.py` — at least 2/3 tests pass
   - [ ] Kaggle notebook runs clean top to bottom
   - [ ] README is accurate and complete
   - [ ] `.env` is not committed (check with `git log --all -- .env`)
   - [ ] `LEARNINGS.md` has an entry for every day
   - [ ] All 5 agents are in the codebase
   - [ ] All 4 skills are in the codebase
   - [ ] `tools/tfidf_matcher.py` exists and is tested
   - [ ] `data/sample_candidates.csv` is committed

3. Fix anything on the checklist that isn't done.

4. Write in `LEARNINGS.md`: What would you build next if you had 2 more weeks? What's the biggest limitation of the current system?

**You're done when:** Every item on the checklist is green. Commit everything.

---

## PHASE 5 — Submission (Days 15–16)

> **Goal:** Submit. Don't add features. Don't refactor. Submit.

---

### Day 15 — Final Kaggle Notebook Polish

**What you're doing:** Polishing the Kaggle notebook for submission. This is not a day to add features. This is a day to make what you have look excellent.

**Tasks:**

1. Read through the entire Kaggle notebook as if you're a judge seeing it for the first time. Ask yourself:
   - Can someone understand what each section does without reading the code?
   - Is every output cell visible (not just code — actual output)?
   - Is the architecture diagram clear?
   - Is the "before vs after" story clear?

2. Add a markdown cell at the top of the notebook with this structure:

```markdown
# HireFlow — Multi-Agent Recruitment Intelligence System

**Track:** Agents for Business
**Author:** Kunal (koi-bito)
**GitHub:** https://github.com/koi-bito/hireflow-capstone

## What This Notebook Demonstrates

| Concept                                  | Where It Appears      |
| ---------------------------------------- | --------------------- |
| Multi-agent system (ADK SequentialAgent) | Section 3 + Section 7 |
| MCP server (Google Search)               | Section 4             |
| Agent skills (4 reusable skills)         | Section 5             |
| Security feature (bias guardrail)        | Section 6             |

**Kaggle badge and certificate will be awarded by end of July 2026.**
```

3. Run the notebook one final time. Make sure every cell has visible output. Don't submit a notebook with empty output cells.

4. Save the notebook.

**You're done when:** The notebook is polished, runs clean, and every cell has output. Don't touch the code after this.

---

### Day 16 — Submit

**What you're doing:** Submitting your project. That's it. No new code.

**Tasks:**

1. Submit the Kaggle notebook to the competition. Deadline is July 6, 2026 at 11:59 PM PT. Submit well before — Kaggle can be slow.

2. Make your final GitHub commit:

```bash
git add .
git commit -m "v1.0 — Kaggle Capstone Submission"
git tag v1.0
git push origin main --tags
```

3. Update your LinkedIn immediately after submitting:

```
Just submitted my Kaggle AI Agents Capstone project — HireFlow, a multi-agent
recruitment intelligence system built on Google ADK.

5 specialized agents (JD Analyst, Bias Guardrail, Sourcing, Matcher, Outreach)
working in sequence via ADK's SequentialAgent. MCP web search for live candidate
sourcing. TF-IDF + LLM semantic scoring. Personalized outreach generation.
Dedicated bias detection safety agent.

Extended my existing AI Talent Scouting project from a Flask TF-IDF script
into a production-style multi-agent system.

GitHub: [link] | Kaggle: [link]

#GoogleADK #AIAgents #Kaggle #MultiAgent #MCP
```

4. Write in `LEARNINGS.md` (final entry): What was the most important thing you learned building this? What would you do differently if you started over?

5. Pin the repo on your GitHub profile. Update your portfolio at `koi-bito.github.io` to include this project.

> 🚀 **Submitted.** You took a Flask TF-IDF script and turned it into a five-agent, MCP-integrated, bias-aware recruitment system in 16 days. That's what engineers do.

---

## Resume Bullet (Update After Submission)

```
Built HireFlow, a multi-agent recruitment system on Google ADK — designed a
5-agent sequential pipeline (JD Analyst, Bias Guardrail, Sourcing, Matcher,
Outreach) using ADK LlmAgent and SequentialAgent, integrated MCP Google Search
for live candidate sourcing, implemented 4 reusable agent skills, and added a
bias detection guardrail using hardcoded pattern matching + LLM audit; extended
an existing TF-IDF Flask project into a production-style agentic system submitted
to the Kaggle AI Agents Capstone (Agents for Business track).
```

---

## Daily Checklist

Use this every single day:

- [ ] Did I commit to GitHub today?
- [ ] Did I write in LEARNINGS.md?
- [ ] Did I understand what I built, or just copy-paste?
- [ ] What is tomorrow's first task?

---

## When You Get Stuck

1. **ADK errors** — search `https://google.github.io/adk-docs/` first. ADK docs are well-written.
2. **Gemini API errors** — check your API key is in `.env` and `load_dotenv()` is called at the top of every script.
3. **MCP not connecting** — verify Node.js is installed (`node --version`) and the MCP server package is installed globally.
4. **Agent not calling the right tool** — rewrite the tool's docstring. The agent uses the docstring to decide when to call a tool. Vague docstrings = wrong tool choices.
5. **SequentialAgent not passing state** — check that each sub-agent's output is in a format the next agent can understand. ADK passes the full conversation history, so agents downstream can see what agents upstream produced.
6. **JSON parsing errors from LLM** — your prompt says "respond only with valid JSON" but the LLM adds backticks anyway. Always strip markdown fences before `json.loads()`. The skills in this plan already handle this.

---

_16 days. 5 agents. 4 skills. 1 MCP integration. 1 bias guardrail. Ship it._
