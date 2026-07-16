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
