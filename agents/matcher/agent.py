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
