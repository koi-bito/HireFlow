# HireFlow — 7-Day Implementation Plan

> **Goal:** Rebuild HireFlow from a broken Flask prototype into a working 5-agent ADK pipeline.
> **Start:** Day 1 | **End:** Day 7
> **After this:** Start the LLM Evaluation Framework immediately.

---

> **How to use this plan:**
> Each day has one clear goal, exact steps, and a "you're done when" checkpoint.
> Keep a `LEARNINGS.md` in the repo. Write 3–5 sentences every day.
> The code in this plan is not copy-paste homework — read it, understand it, then type it yourself. Typing forces understanding. Pasting doesn't.

---

## What You're Building

HireFlow is a multi-agent recruitment system. A recruiter pastes a job description. Five specialized agents handle everything from there.

```
[Job Description Input]
         │
         ▼
┌─────────────────────┐
│   JD Analyst Agent  │  Parses JD → extracts skills, seniority, role type
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Bias Guardrail     │  Flags discriminatory language before anything else runs
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Matcher Agent     │  TF-IDF + LLM semantic scoring on candidate pool
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Outreach Agent    │  Personalized message per shortlisted candidate
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Orchestrator       │  SequentialAgent wiring all four together
└─────────────────────┘
         │
         ▼
[Ranked shortlist + outreach drafts + bias report]
```

**Note on MCP/Sourcing Agent:** The original plan had a 5th sourcing agent using MCP web search. We're skipping it. MCP requires Node.js, a Google Custom Search Engine ID, and extra setup that will eat 2 of your 7 days for marginal gain. The 4-agent pipeline is architecturally complete and more impressive to explain. You can add MCP later in one afternoon if you want.

---

## Final Folder Structure

```
hireflow/
├── agents/
│   ├── jd_analyst/
│   │   └── agent.py
│   ├── bias_guardrail/
│   │   └── agent.py
│   ├── matcher/
│   │   └── agent.py
│   ├── outreach/
│   │   └── agent.py
│   └── orchestrator/
│       └── agent.py
├── skills/
│   ├── parse_jd.py
│   ├── score_candidates.py
│   ├── draft_outreach.py
│   └── detect_bias.py
├── tools/
│   └── tfidf_matcher.py
├── data/
│   └── sample_candidates.csv
├── tests/
│   ├── test_skills.py
│   └── test_agents.py
├── app.py               ← original Flask app, kept for reference, not touched
├── matcher.py           ← original matcher, kept for reference, not touched
├── chatbot.py           ← original chatbot, kept for reference, not touched
├── .env
├── .gitignore
├── requirements.txt
├── LEARNINGS.md
└── README.md
```

---

## Day 1 — Environment + ADK Foundations

**Goal:** Get ADK installed, Gemini API working, and understand the core concepts before writing any HireFlow code.

**Why this day exists:** ADK has specific patterns — how agents are defined, how tools are registered, how runners work. If you skip this and jump to Day 2, you'll spend Day 2 debugging ADK fundamentals instead of building HireFlow.

---

**Step 1 — Create the folder structure**

In your terminal, inside the HireFlow repo:

```bash
mkdir -p agents/jd_analyst agents/bias_guardrail agents/matcher agents/outreach agents/orchestrator
mkdir -p skills tools data tests
touch agents/jd_analyst/agent.py agents/bias_guardrail/agent.py
touch agents/matcher/agent.py agents/outreach/agent.py agents/orchestrator/agent.py
touch skills/parse_jd.py skills/score_candidates.py skills/draft_outreach.py skills/detect_bias.py
touch tools/tfidf_matcher.py tests/test_skills.py tests/test_agents.py
touch LEARNINGS.md requirements.txt .env
```

**Step 2 — Create `.gitignore`**

```
.env
__pycache__/
*.pyc
*.pyo
hireflow_env/
venv/
.venv/
*.db
.DS_Store
```

**Step 3 — Set up virtual environment**

```bash
# Windows
py -3.11 -m venv hireflow_env
hireflow_env\Scripts\activate

# Mac/Linux
python3.11 -m venv hireflow_env
source hireflow_env/bin/activate
```

> Use Python 3.11. ADK has compatibility issues with 3.13.

**Step 4 — Install dependencies**

```bash
pip install google-adk
pip install google-generativeai
pip install python-dotenv
pip install scikit-learn pandas numpy
pip install flask
pip install pytest
```

Create `requirements.txt`:

```
google-adk
google-generativeai
python-dotenv
scikit-learn
pandas
numpy
flask
pytest
```

**Step 5 — Get your Gemini API key**

Go to `https://aistudio.google.com/` → Sign in → "Get API key" → "Create API key" → copy it.

Add to `.env`:

```
GOOGLE_API_KEY=your_key_here
GOOGLE_GENAI_USE_VERTEXAI=FALSE
```

**Step 6 — Verify it works**

Create a temp file `test_setup.py`:

```python
import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-2.0-flash")
response = model.generate_content("Say hello in one sentence.")
print(response.text)
```

Run it: `python test_setup.py`. If you get a response, setup is working. Delete the file after.

**Step 7 — Build your first ADK agent (Hello World)**

This is not HireFlow code. This is you learning the ADK pattern before you need it.

Create a temp file `adk_test.py`:

```python
import os
from dotenv import load_dotenv
load_dotenv()

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# A simple tool
def get_candidate_count() -> dict:
    """Returns the number of candidates in the system."""
    return {"count": 42}

# An agent with that tool
agent = LlmAgent(
    name="test_agent",
    model="gemini-2.0-flash",
    description="A test agent.",
    instruction="Answer questions using your tools.",
    tools=[get_candidate_count]
)

# Runner + session
session_service = InMemorySessionService()
runner = Runner(agent=agent, app_name="test", session_service=session_service)
session = session_service.create_session(app_name="test", user_id="kunal")

# Run it
message = types.Content(role="user", parts=[types.Part(text="How many candidates do we have?")])
for event in runner.run(user_id="kunal", session_id=session.id, new_message=message):
    if event.is_final_response():
        print("Response:", event.response.parts[0].text)
```

Run it. The agent should call `get_candidate_count` automatically and return something like "We currently have 42 candidates in the system."

Understand what happened: you did not tell the agent to call the tool. It read the docstring and decided to call it. That decision process is what makes ADK agents different from regular function calls.

Delete `adk_test.py` after.

**Step 8 — Commit**

```bash
git add .
git commit -m "Day 1: environment setup, ADK verified"
git push
```

**Write in LEARNINGS.md:** What is the difference between an `LlmAgent` and a `SequentialAgent`? When would you use each? Answer in your own words — not copied from the docs.

---

**You're done when:** Gemini API responds, ADK hello world runs, folder structure is on GitHub.

---

## Day 2 — Refactor the TF-IDF Matcher + Sample Data

**Goal:** Take your existing `matcher.py` and turn it into a proper, ADK-compatible tool. Also create the candidate dataset you'll use throughout.

---

**Step 1 — Understand what's changing and why**

Your original `matcher.py` returns a numpy array. ADK tools must return a `dict` — the LLM agent needs to read the output to decide what to do next. A numpy array means nothing to an LLM. A dict with named fields does.

**Step 2 — Create `tools/tfidf_matcher.py`**

```python
"""
TF-IDF Candidate Matcher
Refactored from the original matcher.py to be ADK-compatible.
Returns structured dict instead of raw numpy array.
"""
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def match_candidates_to_jd(job_description: str, candidates_json: str) -> dict:
    """
    Scores a list of candidates against a job description using TF-IDF
    cosine similarity. Returns a ranked list sorted by match score.

    Args:
        job_description: Full text of the job description.
        candidates_json: JSON string of candidates. Each must have
                         'name' (str) and 'skills' (list of str).
                         Example: '[{"name": "Alice", "skills": ["Python", "ML"]}]'

    Returns:
        Dict with 'ranked_candidates' list sorted by match_score descending,
        and 'total' count.
    """
    try:
        candidates = json.loads(candidates_json)
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {str(e)}", "ranked_candidates": []}

    if not candidates:
        return {"error": "Candidates list is empty.", "ranked_candidates": []}

    skills_text = [' '.join(c.get('skills', [])) for c in candidates]

    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([job_description] + skills_text)
    except ValueError as e:
        return {"error": f"Vectorizer error: {str(e)}", "ranked_candidates": []}

    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    ranked = []
    for i, candidate in enumerate(candidates):
        ranked.append({
            "name": candidate.get('name', f'Candidate_{i}'),
            "skills": candidate.get('skills', []),
            "match_score": round(float(similarities[i]), 4),
            "rank": 0
        })

    ranked.sort(key=lambda x: x['match_score'], reverse=True)
    for i, c in enumerate(ranked):
        c['rank'] = i + 1

    return {"ranked_candidates": ranked, "total": len(ranked)}
```

**Step 3 — Test it immediately**

```python
# Run this in a Python shell to verify before moving on
from tools.tfidf_matcher import match_candidates_to_jd
import json

jd = "Python developer with machine learning and NLP experience."
candidates = [
    {"name": "Alice", "skills": ["Python", "machine learning", "NLP", "scikit-learn"]},
    {"name": "Bob", "skills": ["Java", "Spring Boot", "SQL"]},
    {"name": "Priya", "skills": ["Python", "data analysis", "pandas"]}
]

result = match_candidates_to_jd(jd, json.dumps(candidates))
print(json.dumps(result, indent=2))
# Alice should be rank 1, Bob should be rank 3
```

If Alice isn't rank 1, something is wrong. Fix it before continuing.

**Step 4 — Create `data/sample_candidates.csv`**

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

**Step 5 — Write tests for the matcher**

In `tests/test_skills.py`:

```python
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.tfidf_matcher import match_candidates_to_jd


def test_basic_ranking():
    jd = "Python machine learning NLP developer."
    candidates = [
        {"name": "Alice", "skills": ["Python", "machine learning", "NLP"]},
        {"name": "Bob", "skills": ["Java", "Spring Boot"]}
    ]
    result = match_candidates_to_jd(jd, json.dumps(candidates))
    assert result["ranked_candidates"][0]["name"] == "Alice"


def test_empty_candidates():
    result = match_candidates_to_jd("Python developer", json.dumps([]))
    assert "error" in result


def test_invalid_json():
    result = match_candidates_to_jd("Python developer", "not valid json")
    assert "error" in result


def test_all_candidates_returned():
    jd = "Python developer"
    candidates = [
        {"name": "A", "skills": ["Python"]},
        {"name": "B", "skills": ["Java"]},
        {"name": "C", "skills": ["Ruby"]}
    ]
    result = match_candidates_to_jd(jd, json.dumps(candidates))
    assert result["total"] == 3


def test_scores_between_zero_and_one():
    jd = "Python developer"
    candidates = [{"name": "Alice", "skills": ["Python", "Django"]}]
    result = match_candidates_to_jd(jd, json.dumps(candidates))
    for c in result["ranked_candidates"]:
        assert 0.0 <= c["match_score"] <= 1.0
```

Run: `pytest tests/test_skills.py -v` — all 5 tests must pass.

**Step 6 — Commit**

```bash
git add .
git commit -m "Day 2: TF-IDF tool refactored, sample data added, tests passing"
git push
```

**Write in LEARNINGS.md:** Why does the tool return `dict` instead of a numpy array? What would break if you returned the numpy array directly?

---

**You're done when:** All 5 tests pass. Alice ranks #1. Data file is committed.

---

## Day 3 — Build the Four Skills

**Goal:** Write the four skill functions that your agents will use as tools. Skills are just well-typed Python functions — the docstring is what the LLM reads to decide when to call them. Write docstrings like you're explaining the function to a smart person, not a programmer.

---

**Step 1 — `skills/parse_jd.py`**

```python
"""
Skill: Parse Job Description
"""
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def parse_job_description(jd_text: str) -> dict:
    """
    Parses a raw job description and extracts structured recruitment data:
    role title, required skills, preferred skills, seniority level, and role type.
    Use this first, before any candidate matching or sourcing.

    Args:
        jd_text: The full text of the job description.

    Returns:
        Dict with role_title, required_skills, preferred_skills,
        experience_level, role_type, and summary.
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""Extract structured information from this job description.
Respond ONLY with valid JSON. No markdown, no backticks, no explanation.

Job Description:
{jd_text}

Return exactly this structure:
{{
  "role_title": "string",
  "required_skills": ["skill1", "skill2"],
  "preferred_skills": ["skill1", "skill2"],
  "experience_level": "junior|mid|senior",
  "role_type": "technical|non-technical",
  "summary": "one sentence summary"
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
            "error": str(e),
            "role_title": "Unknown",
            "required_skills": [],
            "preferred_skills": [],
            "experience_level": "unknown",
            "role_type": "unknown",
            "summary": "Failed to parse JD"
        }
```

**Step 2 — `skills/detect_bias.py`**

```python
"""
Skill: Bias Detection Guardrail
"""
import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

KNOWN_BIAS_PATTERNS = [
    "young", "energetic team", "digital native", "recent graduate only",
    "native english", "brotherhood", "gentleman", "rockstar", "ninja",
    "culture fit", "must be local", "guys"
]


def detect_bias_in_jd(jd_text: str) -> dict:
    """
    Analyzes a job description for discriminatory or biased language.
    Checks for both hardcoded known-bias patterns and uses an LLM to
    detect subtler forms of exclusionary language. Always run this
    before publishing a JD or sending candidate shortlists.

    Args:
        jd_text: The full text of the job description.

    Returns:
        Dict with bias_detected (bool), severity (none/low/medium/high),
        flags (list of issues), and suggestions (list of fixes).
    """
    flags = []
    jd_lower = jd_text.lower()
    for pattern in KNOWN_BIAS_PATTERNS:
        if pattern in jd_lower:
            flags.append(f"Known bias pattern: '{pattern}'")

    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""You are an HR compliance expert. Analyze this job description for bias.

Job Description:
{jd_text}

Check for: age bias, gender bias, nationality bias, disability exclusion,
culture-fit language that excludes minorities, unnecessarily restrictive requirements.

Respond ONLY with valid JSON. No markdown, no backticks.

Return:
{{
  "additional_flags": ["issue1", "issue2"],
  "severity": "none|low|medium|high",
  "suggestions": ["fix1", "fix2"],
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
            "overall_assessment": f"LLM check failed: {str(e)}. Pattern check only.",
            "error": str(e)
        }
```

**Step 3 — `skills/score_candidates.py`**

```python
"""
Skill: Score and Rank Candidates
Combines TF-IDF matching with LLM semantic scoring.
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
    Scores candidates against a job description using two methods combined:
    TF-IDF cosine similarity for keyword matching and LLM semantic scoring
    for deeper fit analysis. Returns a ranked shortlist with scores and
    reasoning. Use this when you need to identify best-fit candidates.

    Args:
        job_description: Full text of the job description.
        candidates_json: JSON string of candidates. Each needs
                         'name' (str) and 'skills' (list of str).

    Returns:
        Dict with 'shortlist' — top candidates with tfidf_score, llm_score,
        combined_score, and fit_reason.
    """
    tfidf_result = match_candidates_to_jd(job_description, candidates_json)
    if "error" in tfidf_result:
        return tfidf_result

    top_candidates = tfidf_result["ranked_candidates"][:5]

    model = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""You are a senior technical recruiter. Score each candidate's fit.

Job Description:
{job_description}

Top Candidates (by keyword match):
{json.dumps(top_candidates, indent=2)}

For each candidate give:
- llm_score: float 0.0 to 1.0 (1.0 = perfect fit)
- fit_reason: one sentence

Respond ONLY with valid JSON. No markdown, no backticks.

Return:
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

        llm_map = {s["name"]: s for s in llm_scores.get("scored", [])}
        shortlist = []
        for c in top_candidates:
            llm_data = llm_map.get(c["name"], {"llm_score": 0.5, "fit_reason": "Not scored"})
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
```

**Step 4 — `skills/draft_outreach.py`**

```python
"""
Skill: Draft Candidate Outreach
Generates personalized outreach messages per candidate.
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
    Generates a personalized recruiter outreach email for a specific candidate.
    References the candidate's actual skills and explains why they were shortlisted.
    Use this for each candidate after ranking is complete.

    Args:
        candidate_name: Full name of the candidate.
        candidate_skills_json: JSON string of the candidate's skills list.
        role_title: Title of the open position.
        company_name: Name of the hiring company.
        fit_reason: Why this candidate fits (from the scorer output).

    Returns:
        Dict with 'subject' (email subject) and 'message' (email body).
    """
    model = genai.GenerativeModel("gemini-2.0-flash")
    skills = json.loads(candidate_skills_json)

    prompt = f"""Write a personalized recruiter outreach email.

Candidate: {candidate_name}
Skills: {', '.join(skills)}
Role: {role_title}
Company: {company_name}
Why they fit: {fit_reason}

Rules:
- Under 150 words
- Reference 2-3 of their specific skills by name
- Sound human, not corporate
- No "I came across your profile" or "exciting opportunity"
- End with a soft ask (open to a quick chat?) not a formal interview request

Respond ONLY with valid JSON. No markdown, no backticks.

Return:
{{
  "subject": "email subject here",
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
            "subject": f"{role_title} — Would love to connect",
            "message": f"Hi {candidate_name}, your background in {skills[0] if skills else 'your field'} looks like a strong fit for our {role_title} role. Open to a quick chat?",
            "error": str(e)
        }
```

**Step 5 — Test all four skills manually**

Open a Python shell and test each one with a real JD. Don't just run them — look at the output and check if it makes sense. If `parse_jd` returns an empty `required_skills` list for a JD that clearly has requirements, your prompt needs fixing.

**Step 6 — Commit**

```bash
git add .
git commit -m "Day 3: four skills built and tested"
git push
```

**Write in LEARNINGS.md:** Why does the docstring matter so much in ADK tools? What happens if the docstring is vague?

---

**You're done when:** All four skills return sensible output when tested with a real JD.

---

## Day 4 — Build the Four Agents

**Goal:** Wrap each skill in an ADK `LlmAgent`. Each agent is just an agent definition file — no runner logic here, that comes in the orchestrator.

---

**Step 1 — `agents/jd_analyst/agent.py`**

```python
from google.adk.agents import LlmAgent
from skills.parse_jd import parse_job_description
from skills.detect_bias import detect_bias_in_jd

jd_analyst_agent = LlmAgent(
    name="jd_analyst_agent",
    model="gemini-2.0-flash",
    description=(
        "Analyzes job descriptions. Extracts structured requirements "
        "(skills, seniority, role type) and runs an initial bias check. "
        "Always use this first, before any candidate matching."
    ),
    instruction="""You are a senior HR analyst specializing in job description analysis.

When given a job description:
1. Call parse_job_description to extract structured requirements
2. Call detect_bias_in_jd to check for discriminatory language
3. If bias is detected at medium or high severity, highlight it clearly
4. Summarize the key requirements concisely for the recruiter

Be direct. Recruiters are busy. No filler.""",
    tools=[parse_job_description, detect_bias_in_jd]
)
```

**Step 2 — `agents/bias_guardrail/agent.py`**

```python
from google.adk.agents import LlmAgent
from skills.detect_bias import detect_bias_in_jd

bias_guardrail_agent = LlmAgent(
    name="bias_guardrail_agent",
    model="gemini-2.0-flash",
    description=(
        "A compliance agent that audits job descriptions for bias and "
        "discrimination. Always run this before finalizing any recruitment output."
    ),
    instruction="""You are an HR compliance officer focused on fair hiring.

When given a JD:
- Run detect_bias_in_jd
- severity 'none' or 'low': confirm clean with brief notes
- severity 'medium': list each flag with specific rewrite suggestions
- severity 'high': clearly warn this JD must NOT be published as-is

Be specific. Point to exact phrases. Generic advice is useless.""",
    tools=[detect_bias_in_jd]
)
```

**Step 3 — `agents/matcher/agent.py`**

```python
from google.adk.agents import LlmAgent
from skills.score_candidates import score_and_rank_candidates
from tools.tfidf_matcher import match_candidates_to_jd

matcher_agent = LlmAgent(
    name="matcher_agent",
    model="gemini-2.0-flash",
    description=(
        "Scores and ranks candidates against a job description using "
        "TF-IDF keyword matching combined with LLM semantic scoring. "
        "Use this after you have a candidate list and a job description."
    ),
    instruction="""You are a technical recruiter specializing in candidate evaluation.

When given a JD and candidates:
1. Call score_and_rank_candidates to get the ranked shortlist
2. Present the top 3 clearly: rank, combined score, fit reason
3. Flag any candidate with combined_score below 0.3 as a weak match
4. Note the scoring method: TF-IDF + LLM semantic scoring combined

Format as a clean scannable list.""",
    tools=[score_and_rank_candidates, match_candidates_to_jd]
)
```

**Step 4 — `agents/outreach/agent.py`**

```python
from google.adk.agents import LlmAgent
from skills.draft_outreach import draft_outreach_message

outreach_agent = LlmAgent(
    name="outreach_agent",
    model="gemini-2.0-flash",
    description=(
        "Generates personalized outreach emails for shortlisted candidates. "
        "Each message references the candidate's specific skills and explains "
        "why they were chosen. Use after ranking is complete."
    ),
    instruction="""You are an expert recruitment copywriter.

When given a shortlist and role details:
1. Call draft_outreach_message for each candidate in the shortlist
2. Present each message with the candidate name as a clear header
3. If a candidate has no skills listed, note that more profile data is needed

Your output is what a recruiter will send. Make it count.""",
    tools=[draft_outreach_message]
)
```

**Step 5 — Test each agent in isolation**

Test each agent individually before wiring them together. Use this pattern for each:

```python
import os
from dotenv import load_dotenv
load_dotenv()

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from agents.jd_analyst.agent import jd_analyst_agent  # swap per agent

session_service = InMemorySessionService()
runner = Runner(agent=jd_analyst_agent, app_name="test", session_service=session_service)
session = session_service.create_session(app_name="test", user_id="kunal")

# Use a biased JD to test the analyst
jd = """We are looking for a young, energetic Backend Engineer.
Must be a rockstar Python developer. Native English speaker preferred."""

message = types.Content(role="user", parts=[types.Part(text=f"Analyze this JD: {jd}")])
for event in runner.run(user_id="kunal", session_id=session.id, new_message=message):
    if event.is_final_response():
        print(event.response.parts[0].text)
```

The JD Analyst should catch "young", "rockstar", and "Native English speaker" as bias flags. If it doesn't, your `detect_bias_in_jd` skill docstring needs to be clearer.

**Step 6 — Commit**

```bash
git add .
git commit -m "Day 4: four agents built and individually tested"
git push
```

**Write in LEARNINGS.md:** You have two tools in the JD Analyst agent. How does the agent decide which to call first? What would happen if it called them in the wrong order?

---

**You're done when:** All four agents run individually and produce sensible output.

---

## Day 5 — Build the Orchestrator + Full Pipeline Test

**Goal:** Wire all four agents into a `SequentialAgent` orchestrator and run the complete end-to-end pipeline for the first time.

---

**Step 1 — `agents/orchestrator/agent.py`**

```python
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
```

**Step 2 — Run the full pipeline**

Create `run_pipeline.py` in the root:

```python
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
session = session_service.create_session(app_name="hireflow", user_id="recruiter")
message = types.Content(role="user", parts=[types.Part(text=input_text)])

print("Running HireFlow pipeline...\n")
print("=" * 60)

for event in runner.run(user_id="recruiter", session_id=session.id, new_message=message):
    if event.is_final_response():
        print(event.response.parts[0].text)

print("=" * 60)
print("\nPipeline complete.")
```

Run it: `python run_pipeline.py`

**Step 3 — Verify the output has all four stages**

The output should show:
- JD parsed (role title, required skills extracted)
- Bias check result (should be clean for this JD)
- Ranked shortlist (Fatima and Alice should be near the top for an ML role)
- Outreach messages for the top 3

If any stage is missing or the output is garbled, go back to that agent and check the instruction and tool docstrings.

**Step 4 — Commit**

```bash
git add .
git commit -m "Day 5: orchestrator built, full pipeline running end-to-end"
git push
```

**Write in LEARNINGS.md:** What is the difference between `SequentialAgent` and `LlmAgent`? Why is the orchestrator a `SequentialAgent` and not an `LlmAgent`? What would break if you made it an `LlmAgent`?

---

**You're done when:** `python run_pipeline.py` produces output from all four stages without errors.

---

## Day 6 — Tests + Bug Fixes

**Goal:** Write agent tests, run everything, fix what breaks. This is the day that separates "I hacked it together" from "I built it properly."

---

**Step 1 — Add agent tests to `tests/test_agents.py`**

```python
"""
Integration tests for HireFlow agents.
Requires GOOGLE_API_KEY in .env
Run: pytest tests/test_agents.py -v
"""
import json
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
    session = session_service.create_session(app_name="test", user_id="test")
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
```

**Step 2 — Run all tests**

```bash
# Skill tests (fast)
pytest tests/test_skills.py -v

# Agent tests (slower, hits Gemini API)
pytest tests/test_agents.py -v
```

All skill tests must pass. Agent tests should pass most of the time — occasional failures due to LLM non-determinism are acceptable.

**Step 3 — Fix any issues**

Common issues you'll hit and how to fix them:

**"Agent not calling the right tool"** → Rewrite the tool's docstring. Make it more specific about when to call it.

**"JSON parsing error from LLM"** → Check that you're stripping markdown fences before `json.loads()`. All four skills already do this — if you changed the prompt, the LLM might have started wrapping output in backticks again.

**"SequentialAgent not passing state"** → ADK passes the full conversation history between agents. If a downstream agent can't see upstream output, check that the upstream agent's response is in plain text the next agent can read, not buried in a tool result format.

**"Import errors"** → Make sure you're running from the repo root, not from inside a subfolder.

**Step 4 — Commit**

```bash
git add .
git commit -m "Day 6: tests written, bugs fixed, all tests passing"
git push
```

**Write in LEARNINGS.md:** Why do we test skills separately from agents? What makes agent tests harder to write and less reliable than unit tests?

---

**You're done when:** All skill tests pass. At least 3 of 4 agent tests pass.

---

## Day 7 — README + Description + Topics + Done

**Goal:** Make the repo presentable. No new code. Just presentation.

---

**Step 1 — Rewrite `README.md`**

```markdown
# HireFlow — Multi-Agent Recruitment System

A multi-agent recruitment intelligence system built on Google ADK.
Paste a job description and a candidate list — five specialized agents
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
explore Google ADK's agent orchestration patterns — specifically how a
`SequentialAgent` coordinates specialized `LlmAgent` instances, and how
reusable skill functions let agents share capabilities without coupling.
The bias guardrail agent was added as a practical safety layer: discriminatory
JD language is a real problem in recruitment, and catching it before
shortlisting candidates is more useful than catching it after.
```

**Step 2 — Add repo description on GitHub**

Gear icon → About:

```
Multi-agent recruitment system built on Google ADK. JD analysis → bias guardrail → TF-IDF + LLM candidate matching → personalized outreach. SequentialAgent orchestrating 4 specialized LlmAgents.
```

**Step 3 — Add topics**

```
google-adk multi-agent llm recruitment python gemini tfidf scikit-learn ai-agents
```

**Step 4 — Final commit**

```bash
git add .
git commit -m "Day 7: README, description, topics — v1.0 complete"
git push
```

**Write in LEARNINGS.md (final entry):** What is the biggest limitation of the current system? What would you add if you had 2 more weeks?

---

**You're done when:** README is live, repo has description and topics, `python run_pipeline.py` runs clean.

---

## After Day 7

Open `LLM_EVAL_SPEC.md` — the spec we wrote earlier. Start with `evaluator/scorers/base.py`. That's your next project.

HireFlow is done. Move forward.

---

## Daily Checklist

Every single day before you close your laptop:
- [ ] Did I commit to GitHub?
- [ ] Did I write in LEARNINGS.md?
- [ ] Do I understand what I built, or did I just paste code?
- [ ] What is tomorrow's first task?

---

## When You Get Stuck

**ADK errors** → Check `https://google.github.io/adk-docs/` first.

**Gemini API errors** → Check `.env` has the key and `load_dotenv()` is at the top of the file.

**Agent not calling the right tool** → Rewrite the tool's docstring. The LLM uses it to decide when to call the tool. Vague docstring = wrong decisions.

**JSON parsing errors** → The LLM added backticks despite being told not to. All skills handle this — check you haven't removed the fence-stripping logic.

**SequentialAgent not passing context** → ADK passes full conversation history between agents. The issue is usually that an upstream agent's output format is unreadable by the downstream agent. Check the upstream agent's instruction.

**Import errors** → Always run from the repo root, not from inside a subfolder.
```