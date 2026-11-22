# HireFlow Retrieval & Ranking Evaluation Report

**Date:** `2026-11-22 11:18:34 UTC` | **Commit:** `21b22bb` | **Dataset:** 8 JDs x 60 Candidates (480 annotations)

| Retrieval Mode | P@5 | P@10 | R@5 | R@10 | MRR | NDCG@5 | NDCG@10 | Groundedness | Avg Latency (ms) | LLM Calls/Run |
|---|---|---|---|---|---|---|---|---|---|---|
| **keyword** | 1.0000 | 0.7750 | 0.3857 | 0.5683 | 1.0000 | 0.9878 | 0.9517 | 100.0% | 8.9 ms | 0.0 |
| **embedding** | 1.0000 | 0.8250 | 0.3857 | 0.6049 | 1.0000 | 0.9769 | 0.9574 | 100.0% | 32.4 ms | 0.0 |
| **hybrid** | 0.9500 | 0.8000 | 0.3678 | 0.5842 | 0.8750 | 0.8760 | 0.8980 | 100.0% | 31.0 ms | 0.0 |
| **hybrid + LLM** | 0.9750 | 0.8000 | 0.3779 | 0.5842 | 1.0000 | 0.9586 | 0.9421 | 100.0% | 104.1 ms | 1.0 |
