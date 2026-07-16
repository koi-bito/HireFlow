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
