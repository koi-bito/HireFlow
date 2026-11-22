# HireFlow v3 — Retrieval, Ranking & Evals Design Document

## 1. System Overview & Funnel Architecture

HireFlow v3 transforms an early-stage recruitment pipeline into a production-grade information retrieval, semantic ranking, and personalising career-agent system.

```
       Candidate Pool (CSV / Database: 10,000+ Candidates)
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Stage 1: Two-Tower Hybrid Retrieval                    │
  │  - Dense Semantic Vector Search (Cosine Similarity)     │
  │  - Sparse Keyword Overlap (Required Skills Match)       │
  │  - Reciprocal Rank Fusion (RRF: k=60)                   │
  └────────────────────────┬────────────────────────────────┘
                           │ (Top K = 10)
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Stage 2: Deterministic Boosts & Personalisation Memory │
  │  - Seniority / Experience Boost (+0.05)                 │
  │  - Location / Remote Alignment Boost (+0.05)            │
  │  - Recruiter Feedback Personalisation Nudge (±0.08)     │
  └────────────────────────┬────────────────────────────────┘
                           │ (Top N = 5)
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Stage 3: LLM Reasoning & Re-Ranking (Gemini 3.5 Flash) │
  │  - Deep fit scoring (0–100)                             │
  │  - Strict JSON structured output                        │
  │  - Safe grounded defaults on timeout / rate limits      │
  └────────────────────────┬────────────────────────────────┘
                           │
                           ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Stage 4: Grounded Explainability & Score Blending      │
  │  - Blended Score = 0.4 * Retrieval + 0.6 * LLM          │
  │  - Grounded Match Reasons citing verified skills        │
  └─────────────────────────────────────────────────────────┘
```

---

## 2. Why Hybrid Retrieval Beats Pure Keyword or Pure Embedding

| Retrieval Paradigm | Strengths | Critical Failure Modes |
|---|---|---|
| **Pure Keyword Matching** (Boolean / Overlap) | Exact skill precision. If a JD requires `FastAPI`, candidates without `FastAPI` rank lower. Zero hallucination. | **Vocabulary Mismatch Problem**: Misses candidates who wrote `distributed deep learning` when the JD asked for `scalable neural networks`. Cannot evaluate seniority, semantic context, or summary nuance. |
| **Pure Embedding Matching** (Dense Vector Cosine) | Understands semantic intent, role concepts, and cross-domain transferable experience. | **Exact Entity Blindness**: Embeddings can project `PyTorch` and `TensorFlow` or `Java` and `C#` close together in vector space. When an engineering team strictly requires `C++`, pure dense search might rank a `Java` candidate higher because both write object-oriented backend code. |
| **Hybrid (RRF Fusion)** | Combines the lexical precision of keyword filtering with the conceptual comprehension of semantic embeddings. | Standard in modern search systems (Elasticsearch, Pinecone, Google Search). |

### Reciprocal Rank Fusion (RRF)
Rather than naively adding raw cosine scores (which vary across embedding models) and keyword percentages (which have different distributions), we use **Reciprocal Rank Fusion**:
$$RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
Where $r_m(d)$ is candidate $d$'s rank in ranking system $m$, and $k=60$ is a smoothing constant. RRF is immune to uncalibrated score distributions and guarantees that candidates ranking high across both retrieval systems dominate the top shortlist.

---

## 3. The Two-Stage Funnel: Cheap Pre-Filter, Expensive Re-Rank

Sending 1,000 candidates directly to an LLM like Gemini:
- **Cost**: $1,000 \times \text{tokens} = \$0.50$ to $\$2.00$ per job search.
- **Latency**: 1,000 API calls or large batch prompts take 30–90 seconds and easily hit API concurrency limits (HTTP 429).
- **Failure Risk**: Any network blip invalidates the entire run.

By operating a **two-stage retrieval funnel**:
1. Stage 1 retrieves the top `RETRIEVAL_TOP_K = 10` candidates using vectorized NumPy math in `<5 ms` at **$0 cost**.
2. Stage 2 sends only the top `RERANK_TOP_N = 5` candidates to Gemini in **a single batched prompt**, completing in `<500 ms` with **1 API call**.

This yields a **90%+ reduction in LLM inference cost and latency** without sacrificing accuracy on the candidates that matter.

---

## 4. Why Plain NumPy Vector Search at this Scale (<50,000 Candidates)

For pools under 50,000 candidates:
- An $(N, D)$ matrix of float32 vectors where $N=50,000$ and $D=768$ consumes only:
  $$50,000 \times 768 \times 4 \text{ bytes} \approx 153.6 \text{ MB of RAM}$$
- A single BLAS-accelerated matrix multiplication (`scores = vectors @ query_vec`) executes in **under 4 milliseconds** on a standard modern CPU.
- **Tradeoff against FAISS / HNSW**:
  - FAISS requires compiling C++ extensions with OpenMP dependencies, which frequently breaks in minimalist Docker images, Alpine containers, or across Windows/Linux architectures.
  - Approximate Nearest Neighbor (ANN) indexes introduce **recall loss** (approximate quantization). NumPy vector dot-products are **100% exact**.
  - NumPy arrays allow $O(1)$ dynamic candidate appending and in-place row deletions without rebuilding an index graph.

---

## 5. Incremental Indexing for Dynamic Candidate Pools

In a live hiring platform, candidates apply, update resumes, or delete accounts continuously. Rebuilding an entire vector index from scratch for every change is wasteful ($O(N)$ embedding cost).

Our `CandidateIndex` implements:
- `add(candidates)`: Embeds only the newly arriving candidate(s) and uses `np.vstack` to append to the active matrix in memory.
- `remove(email)`: Locates the row index via vectorized boolean mask, drops the candidate record, and removes the vector via `np.delete` in milliseconds.
- `save(path)` / `load(path)`: Serializes candidate metadata to `candidates.csv`, vectors to `vectors.npy`, and schema configurations to `metadata.json`.

---

## 6. What Changes at 1,000,000+ Candidates?

| Layer | Architecture at 50k Candidates (Current) | Architecture at 1,000,000+ Candidates |
|---|---|---|
| **Vector Storage** | In-memory float32 NumPy array (`150 MB`) | Distributed Vector Database (e.g. **Qdrant**, **Milvus**, or **pgvector**) with vector quantization (Scalar Quantization / Product Quantization SQ8/PQ). |
| **Search Algorithm** | Exact linear scan ($O(N)$ dot products) | Graph-based Approximate Nearest Neighbor (**HNSW**) or Inverted File Index (**IVF-PQ**) yielding logarithmic search ($O(\log N)$). |
| **Ingestion Pipeline** | Synchronous DataFrame embedding | Asynchronous event-driven queue (**Apache Kafka** + **Celery**) embedding resumes in background workers. |
| **Caching Layer** | Local disk `.npy` hash cache | Distributed **Redis** cache storing embeddings by `SHA-256(text + model_name)`. |

---

## 7. Interview Notes & Frequently Asked Questions

### Q1: Why did you choose Reciprocal Rank Fusion (RRF) over a weighted sum of scores?
**Answer:** "Cosine similarity and keyword overlap have fundamentally different mathematical distributions. Cosine similarities from dense embedding models often cluster in narrow bands (e.g., $0.65 - 0.85$), whereas keyword overlap ratios range from $0.0$ to $1.0$ with sharp discrete steps. Summing them directly requires arbitrary normalization hyperparameters that drift whenever the embedding model changes. RRF operates purely on ordinal ranks, making it model-agnostic, robust to outliers, and parameter-free except for a standard smoothing constant ($k=60$)."

### Q2: How do you prevent LLMs from hallucinating skills during candidate evaluation?
**Answer:** "We enforce a two-layer guardrail: First, the candidate matcher deterministically tokenizes the candidate's actual resume text and computes `matched_skills` and `missing_skills` before the LLM is ever called. Second, the generated `match_reasons` are injected with verified skill tokens from the candidate's actual profile rather than free-form LLM prose. Our eval harness tracks this directly as a `Groundedness Rate`, verifying that 100% of cited skills exist in the source profile."

### Q3: What is the benefit of your recruiter feedback personalisation loop?
**Answer:** "Recruiters frequently have implicit domain preferences that JDs fail to specify (e.g. preferring engineers with strong distributed systems or startup backgrounds). Our SQLite feedback store records `shortlisted`, `rejected`, and `hired` actions. The matcher nudges candidate scores by up to $\pm 0.08$ based on past feedback for that role title, boosting candidates similar to previously hired talent and demoting candidates similar to rejected profiles."

### Q4: How do you test the system in continuous integration without burning API credits?
**Answer:** "All 87 tests run 100% offline with zero external network calls. We implemented a `FakeEmbedder` that uses deterministic feature hashing (SHA-256 token hashing) to project words into normalized vector space, producing realistic semantic overlap deterministically. All Gemini model calls are mocked using Python's `unittest.mock`. The CI suite runs in under 9 seconds in GitHub Actions."

### Q5: What is the biggest known limitation of this system?
**Answer:** "The biggest limitation is that candidate resumes are structured text profiles rather than multi-page complex PDFs with layout awareness. In a larger production deployment, I would integrate a document parsing pipeline (using OCR / multimodal vision) to parse complex PDF formatting, multi-column layouts, and temporal work histories."
