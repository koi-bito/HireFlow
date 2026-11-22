"""
agents/candidate_matcher.py
----------------------------
Agent 2 - Candidate Matcher (v3: Embeddings, Hybrid Retrieval & LLM Re-Ranking)

Pipeline Stages:
  1. Query Building: Generates structured query text from parsed_jd and embeds it.
  2. Candidate Retrieval:
     - "keyword": Keyword overlap ratio against required skills.
     - "embedding": Dense vector cosine similarity via CandidateIndex.
     - "hybrid": Reciprocal Rank Fusion (RRF) blending dense vector and keyword rank lists.
  3. Soft Filters & Boosts: Experience and location alignment boosts.
  4. LLM Re-Ranking: Top RERANK_TOP_N candidates evaluated with Gemini for deep reasoning.
  5. Final Scoring: Configurable blend of retrieval score and LLM score.
  6. Grounded Explainability: Match reasons grounded in verified matched and missing skills.
"""

import json
import logging
import os
import re
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from google import genai

import config
from pipeline.state import HireFlowState
from retrieval.embedder import Embedder, get_embedder
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.index import CandidateIndex
from retrieval.personalisation import calculate_personalisation_nudge
from storage.db import get_feedback_for_role

load_dotenv()
logger = logging.getLogger(__name__)

_REQUIRED_CSV_COLUMNS = {"name", "email", "skills", "years_experience", "location", "summary"}

_BATCH_SCORING_PROMPT = """You are a senior technical recruiter. Evaluate and score the following candidates for the given role from 0 to 100.
Provide 3 specific match reasons per candidate (referencing their actual skills and experience), and assign a fit flag:
- "strong_match" (score 80-100)
- "partial_match" (score 50-79)
- "weak_match" (score 0-49)

Return ONLY a valid JSON array of objects with no other text, markdown, or explanation:
[
  {{
    "email": "candidate email",
    "score": <integer 0-100>,
    "match_reasons": ["reason 1", "reason 2", "reason 3"],
    "flag": "strong_match" | "partial_match" | "weak_match"
  }}
]

Role requirements:
{parsed_jd}

Candidates to evaluate:
{candidates_text}
"""


def _keyword_overlap(required_skills: list[str], candidate_skills: str) -> float:
    """Computes simple overlap ratio between required skills and candidate skills string."""
    if not required_skills:
        return 1.0
    req = {s.lower().strip() for s in required_skills if s.strip()}
    if not req:
        return 1.0
    cand = {s.lower().strip() for s in str(candidate_skills).split(",") if s.strip()}
    overlap = len(req & cand)
    return overlap / len(req)


def extract_skill_matches(
    required_skills: list[str],
    candidate_skills: str,
    summary: str = "",
) -> tuple[list[str], list[str]]:
    """
    Identifies matched and missing skills using case-insensitive token and substring matching.
    Returns: (matched_skills, missing_skills)
    """
    if not required_skills:
        return ([], [])
    cand_text = f"{candidate_skills} {summary}".lower()
    matched = []
    missing = []
    for skill in required_skills:
        s_clean = skill.strip()
        if not s_clean:
            continue
        pattern = r"\b" + re.escape(s_clean.lower()) + r"\b"
        if re.search(pattern, cand_text) or s_clean.lower() in cand_text:
            matched.append(s_clean)
        else:
            missing.append(s_clean)
    return (matched, missing)


def calculate_experience_boost(required_exp_str: str, candidate_exp: Any) -> float:
    """Applies soft boost if candidate meets or exceeds stated years of experience."""
    try:
        req_match = re.search(r"\d+", str(required_exp_str))
        cand_match = re.search(r"\d+", str(candidate_exp))
        if req_match and cand_match:
            req_years = int(req_match.group())
            cand_years = int(cand_match.group())
            if cand_years >= req_years:
                return config.EXPERIENCE_BOOST_WEIGHT
            elif cand_years >= req_years * 0.7:
                return config.EXPERIENCE_BOOST_WEIGHT * 0.5
    except Exception:
        pass
    return 0.0


def calculate_location_boost(required_location: str, candidate_location: str) -> float:
    """Applies soft boost if candidate location matches JD or either is Remote."""
    req_loc = str(required_location).lower().strip()
    cand_loc = str(candidate_location).lower().strip()
    if not req_loc or not cand_loc:
        return 0.0
    if "remote" in req_loc or "remote" in cand_loc or req_loc in cand_loc or cand_loc in req_loc:
        return config.LOCATION_BOOST_WEIGHT
    return 0.0


def build_jd_query_text(parsed_jd: dict) -> str:
    """Constructs a dense query string from structured job requirements."""
    parts = []
    title = parsed_jd.get("job_title", "")
    req = parsed_jd.get("required_skills", [])
    pref = parsed_jd.get("preferred_skills", [])
    exp = parsed_jd.get("years_of_experience", "")
    loc = parsed_jd.get("location", "")

    if title:
        parts.append(f"Role: {title}")
    if req:
        skills_str = ", ".join(req) if isinstance(req, list) else str(req)
        parts.append(f"Required Skills: {skills_str}")
    if pref:
        pref_str = ", ".join(pref) if isinstance(pref, list) else str(pref)
        parts.append(f"Preferred Skills: {pref_str}")
    if exp:
        parts.append(f"Experience: {exp}")
    if loc:
        parts.append(f"Location: {loc}")

    return " | ".join(parts) if parts else "General Software Role"


def _build_candidate_text(row: pd.Series) -> str:
    """Format candidate profile for LLM prompt."""
    return (
        f"Email: {row['email']}\n"
        f"Name: {row['name']}\n"
        f"Skills: {row['skills']}\n"
        f"Years of Experience: {row['years_experience']}\n"
        f"Location: {row['location']}\n"
        f"Summary: {row['summary']}"
    )


def _parse_llm_json(text: str) -> Any:
    """Extract JSON object or array from LLM response."""
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    return json.loads(cleaned)


# Module-level client, lazily instantiated
_client: Optional[genai.Client] = None


def get_genai_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.GOOGLE_API_KEY)
    return _client


def _score_candidates_llm(
    candidates_to_score: list[pd.Series],
    parsed_jd: dict,
    matched_skills_map: dict[str, list[str]],
    missing_skills_map: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    """
    Calls Gemini to score a batch of candidates.
    Falls back gracefully to grounded heuristic scores on LLM failure.
    """
    if not candidates_to_score:
        return {}

    # Fast offline path when in test mode without an explicit client mock
    if os.getenv("HIREFLOW_TEST_MODE") == "1" and _client is None:
        results_by_email: dict[str, dict[str, Any]] = {}
        for row in candidates_to_score:
            email = row["email"]
            matched = matched_skills_map.get(email, [])
            missing = missing_skills_map.get(email, [])
            overlap = _keyword_overlap(parsed_jd.get("required_skills", []), row["skills"])
            heuristic_score = int(round(overlap * 100))
            flag = "strong_match" if heuristic_score >= 80 else ("partial_match" if heuristic_score >= 50 else "weak_match")
            reasons = [
                f"Demonstrated skills in: {', '.join(matched[:3])}" if matched else "Matches profile keywords",
                f"Missing background in: {', '.join(missing[:2])}" if missing else "Meets required technical criteria",
                f"Experience aligns with {parsed_jd.get('job_title', 'the role')}",
            ]
            results_by_email[email] = {
                "score": heuristic_score,
                "match_reasons": reasons,
                "flag": flag,
            }
        return results_by_email

    formatted_blocks = []
    for c in candidates_to_score:
        formatted_blocks.append(_build_candidate_text(c))
    candidates_text = "\n\n---\n\n".join(formatted_blocks)

    prompt = _BATCH_SCORING_PROMPT.format(
        parsed_jd=json.dumps(parsed_jd, indent=2),
        candidates_text=candidates_text,
    )

    results_by_email: dict[str, dict[str, Any]] = {}

    try:
        client = _client if _client is not None else get_genai_client()
        response = client.models.generate_content(
            model=config.MODEL_NAME,
            contents=prompt,
        )
        parsed = _parse_llm_json(response.text)

        if isinstance(parsed, list):
            for item in parsed:
                email = item.get("email")
                if email:
                    results_by_email[email] = {
                        "score": int(item.get("score", 50)),
                        "match_reasons": item.get("match_reasons", []),
                        "flag": item.get("flag", "partial_match"),
                    }
        elif isinstance(parsed, dict):
            # Single object response fallback (e.g. legacy test mocks)
            for row in candidates_to_score:
                em = row["email"]
                results_by_email[em] = {
                    "score": int(parsed.get("score", 50)),
                    "match_reasons": list(parsed.get("match_reasons", [])),
                    "flag": parsed.get("flag", "partial_match"),
                }
    except Exception as exc:
        logger.warning("Candidate Matcher: LLM scoring failed (%s) - using grounded safe defaults", exc)

    # Fill any missing candidates with grounded heuristic fallback
    for row in candidates_to_score:
        email = row["email"]
        if email not in results_by_email:
            matched = matched_skills_map.get(email, [])
            missing = missing_skills_map.get(email, [])
            overlap = _keyword_overlap(parsed_jd.get("required_skills", []), row["skills"])
            heuristic_score = int(round(overlap * 100))

            if heuristic_score >= 80:
                flag = "strong_match"
            elif heuristic_score >= 50:
                flag = "partial_match"
            else:
                flag = "weak_match"

            reasons = []
            if matched:
                reasons.append(f"Demonstrated skills in: {', '.join(matched[:3])}")
            if missing:
                reasons.append(f"Missing required background in: {', '.join(missing[:2])}")
            else:
                reasons.append("Matches all core technical skill requirements")
            reasons.append(f"Candidate profile aligns with {parsed_jd.get('job_title', 'role')}")

            results_by_email[email] = {
                "score": heuristic_score,
                "match_reasons": reasons,
                "flag": flag,
            }

    return results_by_email


def run_candidate_matcher(
    state: HireFlowState,
    embedder: Optional[Embedder] = None,
) -> dict:
    """
    LangGraph node function for Agent 2 - Candidate Matcher.

    Args:
        state: The current HireFlowState.
        embedder: Optional injected Embedder instance (useful for offline tests).

    Returns:
        Updated keys: 'ranked_candidates' and (optionally) 'errors'.
    """
    errors: list = list(state.get("errors", []))
    parsed_jd: dict = state.get("parsed_jd") or {}
    csv_path: str = state.get("candidates_csv_path", "data/sample_candidates.csv")
    mode: str = getattr(config, "MATCHER_MODE", "hybrid").lower()

    # ── Load Candidate CSV ────────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        msg = f"Candidate Matcher: CSV not found at '{csv_path}' - falling back to sample data."
        logger.warning(msg)
        errors.append(msg)
        try:
            df = pd.read_csv("data/sample_candidates.csv")
        except Exception as exc:
            err = f"Candidate Matcher: cannot load candidate data - {exc}"
            errors.append(err)
            return {"ranked_candidates": [], "errors": errors}
    except Exception as exc:
        err = f"Candidate Matcher: error reading CSV - {exc}"
        errors.append(err)
        return {"ranked_candidates": [], "errors": errors}

    # ── Validate & Clean Data ─────────────────────────────────────────────────
    missing_cols = _REQUIRED_CSV_COLUMNS - set(df.columns)
    if missing_cols:
        err = f"Candidate Matcher: CSV missing required columns: {missing_cols}"
        errors.append(err)
        logger.error(err)
        return {"ranked_candidates": [], "errors": errors}

    df = df.dropna(how="all").fillna("")
    if df.empty:
        errors.append("Candidate Matcher: candidate pool is empty.")
        return {"ranked_candidates": [], "errors": errors}

    # Deduplicate by email
    df = df.drop_duplicates(subset=["email"], keep="first").reset_index(drop=True)

    # Truncate overly long summaries for embedding / prompt safety
    df["summary"] = df["summary"].apply(lambda s: str(s)[:1000])

    required_skills: list[str] = parsed_jd.get("required_skills", [])
    required_exp: str = parsed_jd.get("years_of_experience", "")
    required_loc: str = parsed_jd.get("location", "")

    # Pre-extract skill matches & missing skills for all candidates
    matched_skills_map: dict[str, list[str]] = {}
    missing_skills_map: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        email = row["email"]
        matched, missing = extract_skill_matches(required_skills, row["skills"], row["summary"])
        matched_skills_map[email] = matched
        missing_skills_map[email] = missing

    # ── Step 1 & 2: Retrieval ─────────────────────────────────────────────────
    keyword_scores: dict[str, float] = {}
    for _, row in df.iterrows():
        keyword_scores[row["email"]] = _keyword_overlap(required_skills, row["skills"])

    dense_scores: dict[str, float] = {}
    active_mode = mode

    if active_mode in {"embedding", "hybrid"}:
        try:
            active_embedder = embedder or get_embedder()
            index = CandidateIndex(embedder=active_embedder)
            index.build_from_dataframe(df)

            query_text = build_jd_query_text(parsed_jd)
            query_vec = active_embedder.embed_texts([query_text])[0]
            search_results = index.search(query_vec, top_k=len(df))

            for cand_dict, cos_sim in search_results:
                dense_scores[cand_dict["email"]] = float(cos_sim)
        except Exception as exc:
            msg = f"Candidate Matcher: Embedding service failed ({exc}) - falling back to keyword mode."
            logger.warning(msg)
            errors.append(msg)
            active_mode = "keyword"

    # Compute candidate retrieval score based on active mode
    retrieval_scores: dict[str, float] = {}

    if active_mode == "keyword":
        retrieval_scores = keyword_scores
    elif active_mode == "embedding":
        retrieval_scores = dense_scores
    elif active_mode == "hybrid":
        # Reciprocal Rank Fusion
        keyword_ranked = sorted(keyword_scores.keys(), key=lambda em: keyword_scores[em], reverse=True)
        dense_ranked = sorted(dense_scores.keys(), key=lambda em: dense_scores.get(em, 0.0), reverse=True)
        rrf_scores = reciprocal_rank_fusion([keyword_ranked, dense_ranked], k=config.RRF_K)
        retrieval_scores = rrf_scores

    # ── Step 3: Hard Filters, Soft Boosts & Personalisation ───────────────────
    role_title = parsed_jd.get("job_title", "")
    try:
        past_feedback = get_feedback_for_role(role_title)
    except Exception:
        past_feedback = []

    boosted_retrieval_scores: dict[str, float] = {}
    personalisation_notes: dict[str, str] = {}

    for _, row in df.iterrows():
        em = row["email"]
        base_retrieval = retrieval_scores.get(em, 0.0)
        exp_boost = calculate_experience_boost(required_exp, row["years_experience"])
        loc_boost = calculate_location_boost(required_loc, row["location"])

        nudge, p_note = calculate_personalisation_nudge(
            candidate_email=em,
            candidate_profile_text=row["summary"],
            candidate_vector=None,
            past_feedback=past_feedback,
        )
        if p_note:
            personalisation_notes[em] = p_note

        total_score = min(1.0, max(0.0, base_retrieval + exp_boost + loc_boost + nudge))
        boosted_retrieval_scores[em] = total_score

    # Top candidates for retrieval stage
    sorted_by_retrieval = sorted(
        df["email"].tolist(),
        key=lambda em: boosted_retrieval_scores.get(em, 0.0),
        reverse=True,
    )
    retrieved_emails = sorted_by_retrieval[: config.RETRIEVAL_TOP_K]
    df_retrieved = df[df["email"].isin(retrieved_emails)].copy()

    # ── Step 4: LLM Re-Rank (Top N) ──────────────────────────────────────────
    rerank_emails = retrieved_emails[: config.RERANK_TOP_N]
    df_to_rerank = [row for _, row in df_retrieved.iterrows() if row["email"] in rerank_emails]

    llm_results = _score_candidates_llm(
        df_to_rerank,
        parsed_jd,
        matched_skills_map,
        missing_skills_map,
    )

    # ── Step 5 & 6: Final Score Blend & Explainability ────────────────────────
    final_ranked: list[dict[str, Any]] = []

    for _, row in df_retrieved.iterrows():
        email = row["email"]
        ret_score = boosted_retrieval_scores.get(email, 0.0)

        if email in llm_results:
            llm_info = llm_results[email]
            llm_score = llm_info["score"]
            llm_reasons = llm_info.get("match_reasons", [])
            flag = llm_info.get("flag", "partial_match")
        else:
            # For candidates retrieved in top K but not sent to LLM re-rank (if RETRIEVAL_TOP_K > RERANK_TOP_N)
            llm_score = int(round(ret_score * 100))
            llm_reasons = []
            flag = "strong_match" if llm_score >= 80 else ("partial_match" if llm_score >= 50 else "weak_match")

        # Blend formula: WEIGHT_RETRIEVAL * (retrieval_score * 100) + WEIGHT_LLM * llm_score
        blended_score = int(round(
            config.WEIGHT_RETRIEVAL * (ret_score * 100) + config.WEIGHT_LLM * llm_score
        ))
        blended_score = max(0, min(100, blended_score))

        # Build explainable match reasons referencing ground truth skills
        matched = matched_skills_map.get(email, [])
        missing = missing_skills_map.get(email, [])

        reasons = []
        if matched:
            reasons.append(f"Demonstrated skills: {', '.join(matched[:3])}")
        if missing:
            reasons.append(f"Missing core requirements: {', '.join(missing[:2])}")
        elif not missing and matched:
            reasons.append("Matches all core technical skill requirements")

        # Include model's qualitative reason if available
        if llm_reasons:
            reasons.append(llm_reasons[0])
        elif not reasons:
            reasons.append("General profile alignment with job requirements")

        if email in personalisation_notes:
            reasons.append(personalisation_notes[email])

        # Ensure exactly 3 reasons
        while len(reasons) < 3:
            reasons.append("Profile verified against role criteria")
        reasons = reasons[:3]

        final_ranked.append({
            "name": row["name"],
            "email": email,
            "score": blended_score,
            "flag": flag,
            "match_reasons": reasons,
            # Additive fields
            "retrieval_score": round(ret_score, 4),
            "llm_score": llm_score,
            "matched_skills": matched,
            "missing_skills": missing,
        })

    # Sort final output by blended score descending
    final_ranked.sort(key=lambda x: x["score"], reverse=True)
    logger.info("Candidate Matcher: ranked %d candidates (mode=%s).", len(final_ranked), active_mode)

    return {"ranked_candidates": final_ranked, "errors": errors}
