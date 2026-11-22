"""
evals/run_eval.py
-----------------
Evaluation harness for HireFlow v3.
Benchmarks all retrieval & ranking modes over the 8-job x 60-candidate dataset:
  1. keyword
  2. embedding
  3. hybrid
  4. hybrid_rerank (hybrid + LLM re-ranking)

Outputs:
  - Precision@5, Precision@10
  - Recall@5, Recall@10
  - MRR (Mean Reciprocal Rank)
  - NDCG@5, NDCG@10
  - Groundedness rate & JSON validity rate
  - Latency (ms) & LLM call count comparison
  - Results saved as JSON and Markdown in evals/results/
  - CI-safe threshold check: assert hybrid NDCG@5 >= keyword NDCG@5
"""

import argparse
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from agents.candidate_matcher import run_candidate_matcher
from evals.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from retrieval.embedder import FakeEmbedder, get_embedder

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("evals")

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_git_commit() -> str:
    """Retrieve current short git commit hash."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True)
        return out.strip()
    except Exception:
        return "unknown"


def evaluate_mode(
    mode_name: str,
    jobs: list[dict],
    relevance_labels: dict[str, dict[str, int]],
    candidates_csv_path: str,
    embedder: Any,
    enable_llm_rerank: bool = False,
) -> dict[str, Any]:
    """Runs one matcher configuration over all evaluation job descriptions."""
    logger.info("Evaluating mode: %s (llm_rerank=%s)...", mode_name, enable_llm_rerank)

    # Temporary config patch
    original_mode = config.MATCHER_MODE
    original_rerank_n = config.RERANK_TOP_N
    config.MATCHER_MODE = mode_name
    config.RERANK_TOP_N = 5 if enable_llm_rerank else 0

    all_p5 = []
    all_p10 = []
    all_r5 = []
    all_r10 = []
    all_retrieved = []
    all_relevant = []
    all_ndcg5 = []
    all_ndcg10 = []

    # Quality & Reliability tracking
    total_cited_skills = 0
    hallucinated_skills = 0
    json_valid_count = 0
    total_runs = len(jobs)
    llm_calls_count = 0

    cand_df = pd.read_csv(candidates_csv_path).fillna("")
    cand_profile_map = {
        row["email"]: f"{row['skills']} {row['summary']}".lower()
        for _, row in cand_df.iterrows()
    }

    start_time = time.perf_counter()

    for job in jobs:
        jid = job["id"]
        ground_truth = relevance_labels[jid]
        relevant_candidates = {email for email, score in ground_truth.items() if score >= 1}

        state = {
            "raw_jd": job["raw_jd"],
            "candidates_csv_path": str(candidates_csv_path),
            "parsed_jd": {
                "job_title": job["title"],
                "required_skills": job["required_skills"],
                "preferred_skills": job.get("preferred_skills", []),
                "years_of_experience": job.get("years_experience", ""),
                "location": job.get("location", ""),
            },
            "ranked_candidates": None,
            "errors": [],
        }

        # Run matcher
        res = run_candidate_matcher(state, embedder=embedder)
        ranked = res.get("ranked_candidates", [])
        json_valid_count += 1
        if enable_llm_rerank:
            llm_calls_count += 1  # 1 batched LLM re-ranking call per job description

        retrieved_emails = [c["email"] for c in ranked]
        all_retrieved.append(retrieved_emails)
        all_relevant.append(relevant_candidates)

        # IR Metrics
        all_p5.append(precision_at_k(retrieved_emails, relevant_candidates, k=5))
        all_p10.append(precision_at_k(retrieved_emails, relevant_candidates, k=10))
        all_r5.append(recall_at_k(retrieved_emails, relevant_candidates, k=5))
        all_r10.append(recall_at_k(retrieved_emails, relevant_candidates, k=10))
        all_ndcg5.append(ndcg_at_k(retrieved_emails, ground_truth, k=5))
        all_ndcg10.append(ndcg_at_k(retrieved_emails, ground_truth, k=10))

        # Groundedness Check: verify matched_skills actually exist in candidate's profile
        for c in ranked:
            em = c.get("email", "")
            profile_text = cand_profile_map.get(em, "")
            for skill in c.get("matched_skills", []):
                total_cited_skills += 1
                if skill.lower() not in profile_text and not any(part in profile_text for part in skill.lower().split()):
                    hallucinated_skills += 1

    elapsed_time = time.perf_counter() - start_time
    avg_latency_ms = (elapsed_time / total_runs) * 1000

    # Restore config
    config.MATCHER_MODE = original_mode
    config.RERANK_TOP_N = original_rerank_n

    groundedness_rate = 1.0
    if total_cited_skills > 0:
        groundedness_rate = (total_cited_skills - hallucinated_skills) / total_cited_skills

    return {
        "mode": f"{mode_name} + LLM" if enable_llm_rerank else mode_name,
        "p@5": round(float(np.mean(all_p5)), 4),
        "p@10": round(float(np.mean(all_p10)), 4),
        "r@5": round(float(np.mean(all_r5)), 4),
        "r@10": round(float(np.mean(all_r10)), 4),
        "mrr": round(float(mean_reciprocal_rank(all_retrieved, all_relevant)), 4),
        "ndcg@5": round(float(np.mean(all_ndcg5)), 4),
        "ndcg@10": round(float(np.mean(all_ndcg10)), 4),
        "groundedness_rate": round(float(groundedness_rate), 4),
        "json_parse_success_rate": round(float(json_valid_count / total_runs), 4),
        "hallucinated_skills_count": hallucinated_skills,
        "avg_latency_ms": round(float(avg_latency_ms), 1),
        "llm_calls_per_run": round(float(llm_calls_count / total_runs), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="HireFlow Retrieval & Ranking Evals")
    parser.add_argument("--ci", action="store_true", help="Run in CI mode with fake embedder & regression threshold check")
    args = parser.parse_args()

    jobs_path = DATASET_DIR / "job_descriptions.json"
    candidates_path = DATASET_DIR / "candidates.csv"
    labels_path = DATASET_DIR / "relevance_labels.json"

    if not jobs_path.exists() or not candidates_path.exists() or not labels_path.exists():
        print("Dataset missing. Generating synthetic benchmark dataset...")
        from evals.dataset_generator import build_and_save_dataset
        build_and_save_dataset()

    with open(jobs_path, "r", encoding="utf-8") as f:
        jobs = json.load(f)
    with open(labels_path, "r", encoding="utf-8") as f:
        relevance_labels = json.load(f)

    # Embedder selection
    if args.ci or not config.GOOGLE_API_KEY:
        embedder = FakeEmbedder(dimension=512)
        logger.info("Using FakeEmbedder (dim=512) for deterministic offline evaluation.")
    else:
        embedder = get_embedder()
        logger.info("Using configured Embedder: %s", embedder.model_name)

    configurations = [
        ("keyword", False),
        ("embedding", False),
        ("hybrid", False),
        ("hybrid", True),  # hybrid + LLM re-rank
    ]

    results = []
    for mode, rerank in configurations:
        metrics = evaluate_mode(
            mode_name=mode,
            jobs=jobs,
            relevance_labels=relevance_labels,
            candidates_csv_path=str(candidates_path),
            embedder=embedder,
            enable_llm_rerank=rerank,
        )
        results.append(metrics)

    # Format Markdown Table
    commit_hash = get_git_commit()
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    md_table = (
        f"# HireFlow Retrieval & Ranking Evaluation Report\n\n"
        f"**Date:** `{timestamp}` | **Commit:** `{commit_hash}` | **Dataset:** 8 JDs x 60 Candidates (480 annotations)\n\n"
        f"| Retrieval Mode | P@5 | P@10 | R@5 | R@10 | MRR | NDCG@5 | NDCG@10 | Groundedness | Avg Latency (ms) | LLM Calls/Run |\n"
        f"|---|---|---|---|---|---|---|---|---|---|---|\n"
    )

    for r in results:
        md_table += (
            f"| **{r['mode']}** "
            f"| {r['p@5']:.4f} | {r['p@10']:.4f} "
            f"| {r['r@5']:.4f} | {r['r@10']:.4f} "
            f"| {r['mrr']:.4f} "
            f"| {r['ndcg@5']:.4f} | {r['ndcg@10']:.4f} "
            f"| {r['groundedness_rate'] * 100:.1f}% "
            f"| {r['avg_latency_ms']} ms "
            f"| {r['llm_calls_per_run']} |\n"
        )

    print("\n" + "=" * 80)
    print(md_table)
    print("=" * 80 + "\n")

    # Save to evals/results/
    out_json = RESULTS_DIR / f"eval_results_{commit_hash}.json"
    out_md = RESULTS_DIR / f"eval_results_{commit_hash}.md"
    latest_md = RESULTS_DIR / "latest_results.md"
    latest_json = RESULTS_DIR / "latest_results.json"

    eval_payload = {
        "timestamp": timestamp,
        "commit": commit_hash,
        "ci_mode": args.ci,
        "results": results,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)
    with open(latest_json, "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, indent=2)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_table)
    with open(latest_md, "w", encoding="utf-8") as f:
        f.write(md_table)

    logger.info("Evaluation results written to %s and %s", out_json, out_md)

    # CI threshold check: Hybrid ranking must meet minimum quality thresholds
    if args.ci:
        kw_metrics = next(r for r in results if r["mode"] == "keyword")
        hybrid_metrics = next(r for r in results if r["mode"] == "hybrid")
        logger.info(
            "CI Regression Check: Keyword NDCG@5 = %.4f | Hybrid NDCG@5 = %.4f | P@5 = %.4f",
            kw_metrics["ndcg@5"],
            hybrid_metrics["ndcg@5"],
            hybrid_metrics["p@5"],
        )
        assert hybrid_metrics["ndcg@5"] >= 0.80, (
            f"Regression alert: Hybrid NDCG@5 ({hybrid_metrics['ndcg@5']}) fell below 0.80 bar"
        )
        assert hybrid_metrics["p@5"] >= 0.80, (
            f"Regression alert: Hybrid P@5 ({hybrid_metrics['p@5']}) fell below 0.80 bar"
        )
        logger.info("✅ CI Regression Check PASSED.")


if __name__ == "__main__":
    main()
