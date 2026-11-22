# HireFlow Learnings

## Day 1
**What is the difference between an `LlmAgent` and a `SequentialAgent`? When would you use each?**

An `LlmAgent` is a single specialised "brain" powered by a large language model. It receives an instruction and a clear input, runs one focused task (like parsing a job description or scoring a candidate), and returns structured output. You use an `LlmAgent` when you need deep reasoning on a narrow, well-defined job.

A `SequentialAgent` is an orchestrator, not a thinker. It does not reason itself - it enforces a strict, deterministic execution order across multiple `LlmAgent`s. In HireFlow, the LangGraph `StateGraph` acts as our `SequentialAgent`: it passes the shared `HireFlowState` through JD Parser -> Candidate Matcher -> Outreach Composer -> Interview Scheduler in a guaranteed order. You use a `SequentialAgent` when the output of one specialised agent must become the input for the next, with no ambiguity about sequence.

## Day 2
**Why do we use a shared `TypedDict` (HireFlowState) instead of passing arguments directly between agents?**

LangGraph's `StateGraph` requires a single shared state object that all nodes read from and write to. Using a `TypedDict` (`HireFlowState`) instead of raw function arguments gives us three benefits: (1) every agent has full visibility of everything that has happened so far in the pipeline - not just what was passed to it; (2) it makes the data contract explicit and type-safe, so mistakes like a missing key are caught early; and (3) each agent only returns the keys it modifies, keeping side-effects local and making debugging straightforward via the `errors` list that every agent appends to.

## Day 3
**Why does the JD Parser instruct Gemini to return "ONLY valid JSON"? What breaks if you skip that instruction?**

LLMs, by default, are conversational - they will wrap answers in explanatory prose ("Here is the JSON you requested: ..."). If we parse the raw response with `json.loads()` without constraining the output format, it will throw a `JSONDecodeError` because the string is not valid JSON. By explicitly instructing the model to return ONLY valid JSON with no other text, markdown, or explanation, we make the output directly machine-readable. We still wrap the call in a `try/except` with a safe default, because even with the instruction, the model can occasionally fail - graceful degradation matters more than assuming perfect LLM compliance.

## Day 4
**Why does the Candidate Matcher use a two-step scoring approach (keyword pre-filter then LLM scoring) instead of sending all candidates straight to Gemini?**

Sending every candidate directly to Gemini for scoring would be slow and expensive, especially with large candidate pools. The keyword overlap pre-filter is a cheap, instant computation - it compares `required_skills` from the parsed JD against each candidate's skills field using a simple set intersection ratio. Only candidates who pass this fast check are then sent to the LLM for a deeper, more expensive 0-100 fit score with match reasons. This two-step pattern (cheap filter first, expensive model second) is a standard production pattern for keeping AI pipelines cost-efficient and fast without sacrificing quality on the candidates that actually matter.

## Day 5
**The Interview Scheduler does not use the LLM to generate interview dates. Why not?**

Dates are deterministic - there is one correct set of next business days, and it can be computed precisely with Python's `datetime` module. Asking an LLM to generate dates is a well-known source of hallucinations: models can confidently produce wrong weekdays, skip weekends incorrectly, or drift from the current date. The Interview Scheduler uses `datetime` to compute real, accurate business-day slots and reserves the LLM only for what it is actually good at - generating a brief, personalised scheduling note per candidate. This is a key architectural principle: use LLMs for language and reasoning tasks; use deterministic code for anything that has a single correct answer.

## Day 6
**Why do our tests mock the Gemini API calls instead of calling the real API?**

Unit and integration tests that call a live LLM would be slow (network latency on every test), expensive (API credits consumed per run), non-deterministic (the model may phrase output differently each time, making assertions brittle), and impossible to run in CI without a secret key. By mocking `_client.models.generate_content` with controlled responses, our tests run in under 8 seconds with zero API calls, zero cost, and 100% reproducibility. The mocks also let us simulate failure cases - like a `JSONDecodeError` or an `Exception("API quota exceeded")` - that would be hard to trigger reliably against a real API. The goal of tests is to validate our code logic, not the LLM's output.

## Day 7
**What is the biggest limitation of the current system? What would you add if you had 2 more weeks?**

The biggest limitation was originally that the pipeline was stateless and single-run - results were held in memory and lost on server restart, and there was no way to resume a failed run or track historical results. If I had two more weeks, I would: (1) add a proper database (PostgreSQL or SQLite) to persist run results so `/results/{run_id}` survives restarts; (2) integrate a real candidate sourcing step that searches LinkedIn or GitHub instead of relying on a static CSV; (3) add a human-in-the-loop approval gate before outreach emails are dispatched; and (4) add streaming support to the FastAPI `/run` endpoint so the client can see agent-by-agent progress in real time rather than waiting for the full pipeline to complete.

## Day 8
**Why do we use dense vector embeddings for candidate retrieval instead of relying solely on keyword matching?**

Keyword matching suffers from the classic "vocabulary mismatch" problem in information retrieval: if a job description asks for "distributed deep learning systems" and a candidate resume highlights "scalable multi-GPU neural network training", keyword overlap yields a score of 0.0 even though the candidate is exceptionally qualified. Dense vector embeddings map job requirements and candidate profiles into a shared geometric semantic space where semantically synonymous phrases cluster together. At the same time, we exclude candidate names from profile embeddings to eliminate demographic and gender bias during vector search.

## Day 9
**Why does Reciprocal Rank Fusion (RRF) work better than a linear weighted sum of keyword and embedding scores?**

Linear score combination ($S = \alpha \cdot S_{\text{keyword}} + (1 - \alpha) \cdot S_{\text{embed}}$) requires raw scores to share the same probability distribution and scale. However, cosine similarity scores from modern transformer embeddings typically cluster within narrow bands (e.g. $0.65 - 0.85$), whereas keyword overlap ratios range from $0.0$ to $1.0$ with discrete step jumps. Reciprocal Rank Fusion ($RRF(d) = \sum \frac{1}{k + r(d)}$) operates purely on the relative ordinal positions (ranks) rather than raw magnitudes. It is completely model-agnostic, requires no calibration when the embedding model is updated, and penalizes candidates that only perform well in one modality while rewarding candidates that rank near the top across both.

## Day 10
**Why do we evaluate candidate ranking with NDCG and MRR instead of simple Accuracy or F1 score?**

Accuracy and F1 are classification metrics: they treat candidate selection as an unordered set and ignore position. In recruitment, order is paramount: a recruiter will only review the top 5 to 10 candidates. If a stellar candidate is retrieved at position #1 versus position #10, classification Accuracy treats both equally, but NDCG (Normalized Discounted Cumulative Gain) applies a logarithmic discount penalty ($\frac{1}{\log_2(\text{rank} + 1)}$) to reflect that lower positions receive drastically less human attention. Furthermore, MRR (Mean Reciprocal Rank) evaluates how quickly the recruiter encounters their very first qualified hire.

## Day 11
**How does recruiter feedback personalization improve matching, and why must the nudge be capped?**

Recruiters frequently possess tacit domain heuristics that are absent from written job descriptions (e.g. preferring engineers from high-velocity startups or with specific architectural disciplines). By storing recruiter actions (`shortlisted`, `rejected`, `hired`) in SQLite, the matcher nudges candidate scores by comparing applicants against past feedback for that role title. However, the nudge must be strictly capped (at $\pm 0.08$) to prevent filter bubbles and catastrophic drift: feedback should provide subtle personalization, not override objective qualification criteria or exclude candidates who meet every hard requirement on paper.

## Day 12
**Why use an agentic function-calling loop for recruiter interactions instead of a static prompt template?**

A static chat prompt can only provide generic answers based on the prompt's pre-loaded text. In contrast, an agentic loop equipped with typed tools (`parse_job_description`, `search_candidates`, `explain_match`, `record_feedback`, `draft_outreach`) allows the LLM to inspect live state, query SQLite databases, execute real Python algorithms, and record persistent mutations. Coupled with conversational turn compaction and long-term feedback memory, the agent operates as an autonomous talent partner capable of taking multi-step actions on behalf of the hiring team.
