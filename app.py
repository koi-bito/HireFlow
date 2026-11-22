"""
app.py
------
FastAPI entry point for the HireFlow multi-agent recruitment pipeline.

Endpoints:
  GET  /health            - Health check (DB + system status)
  POST /run               - Run the multi-agent pipeline
  GET  /results/{run_id}  - Retrieve past run output (survives restarts via SQLite)
  POST /feedback          - Record recruiter feedback (shortlisted / rejected / hired)
  GET  /runs              - List past pipeline runs with pagination
  POST /chat              - Interactive Recruiter Assistant with tool use & memory
"""

import base64
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import config
import storage.db as db
from pipeline.graph import pipeline

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# Initialize database tables and indexes on import
db.init_db()

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HireFlow",
    description="Multi-agent recruitment pipeline: JD Parsing → Hybrid Retrieval → Outreach → Scheduling",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory run store maintained as immediate cache/fallback
_run_store: dict[str, dict] = {}


# ── Request / Response models ──────────────────────────────────────────────────

class RunRequest(BaseModel):
    job_description: str
    candidates_csv: Optional[str] = None        # base64-encoded CSV content
    candidates_csv_path: Optional[str] = None   # direct filesystem path (for local dev)


class FeedbackRequest(BaseModel):
    candidate_email: str
    job_title: str
    action: str = Field(..., description="Action taken: 'shortlisted', 'rejected', or 'hired'")
    run_id: Optional[str] = None
    candidate_summary: Optional[str] = ""
    candidate_skills: Optional[str] = ""


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    database: str
    embedding_model: str


# ── Helper ─────────────────────────────────────────────────────────────────────

def _resolve_csv_path(request: RunRequest) -> str:
    """Determine the CSV path to use, safely writing base64 content if provided."""
    if request.candidates_csv_path and os.path.exists(request.candidates_csv_path):
        return request.candidates_csv_path

    if request.candidates_csv:
        try:
            csv_bytes = base64.b64decode(request.candidates_csv)
            # Input validation: check max file size (e.g. 10MB)
            if len(csv_bytes) > 10 * 1024 * 1024:
                raise HTTPException(status_code=422, detail="Candidate CSV exceeds 10MB limit.")

            tmp_path = f"data/uploaded_{uuid.uuid4().hex[:8]}.csv"
            os.makedirs("data", exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(csv_bytes)
            logger.info("Wrote uploaded CSV to %s", tmp_path)
            return tmp_path
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("Could not decode base64 CSV - using sample. Error: %s", exc)

    return "data/sample_candidates.csv"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Health status check confirming API, SQLite database, and embedding configuration."""
    db_status = "ok"
    try:
        with db.get_connection() as conn:
            conn.execute("SELECT 1;").fetchone()
    except Exception as exc:
        db_status = f"unhealthy: {exc}"

    return HealthResponse(
        status="ok" if db_status == "ok" else "degraded",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="3.0.0",
        database=db_status,
        embedding_model=config.EMBEDDING_MODEL,
    )


@app.post("/run", tags=["Pipeline"])
def run_pipeline(request: RunRequest):
    """
    Run the full HireFlow multi-agent pipeline:
    1. JD Parser (Gemini structured extraction)
    2. Candidate Matcher (Hybrid dense vector + keyword RRF retrieval & LLM re-rank)
    3. Outreach Composer (Personalised outreach generation)
    4. Interview Scheduler (Deterministic business day slot generation)
    """
    raw_jd = request.job_description.strip()
    if not raw_jd:
        raise HTTPException(status_code=422, detail="job_description cannot be empty.")

    # Input validation: size limit on JD (max 50,000 chars)
    if len(raw_jd) > 50000:
        raise HTTPException(status_code=422, detail="job_description exceeds maximum length of 50,000 characters.")

    csv_path = _resolve_csv_path(request)
    run_id = str(uuid.uuid4())

    initial_state = {
        "raw_jd": raw_jd,
        "candidates_csv_path": csv_path,
        "parsed_jd": None,
        "ranked_candidates": None,
        "outreach_messages": None,
        "schedule": None,
        "errors": [],
    }

    logger.info("Starting pipeline run %s", run_id)
    try:
        final_state = pipeline.invoke(initial_state)
    except Exception as exc:
        logger.exception("Pipeline run %s failed: %s", run_id, exc)
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    result = {
        "run_id": run_id,
        "parsed_jd": final_state.get("parsed_jd"),
        "ranked_candidates": final_state.get("ranked_candidates", []),
        "outreach_messages": final_state.get("outreach_messages", []),
        "schedule": final_state.get("schedule", []),
        "errors": final_state.get("errors", []),
    }

    # Persist in memory and in SQLite
    _run_store[run_id] = result
    job_title = (result.get("parsed_jd") or {}).get("job_title", "Unknown Role")
    try:
        db.save_run(
            run_id=run_id,
            result=result,
            job_title=job_title,
            raw_jd=raw_jd,
            candidates_csv_path=csv_path,
        )
    except Exception as exc:
        logger.warning("Could not persist run %s to SQLite: %s", run_id, exc)

    logger.info("Pipeline run %s completed. Errors: %s", run_id, result["errors"])
    return result


@app.get("/results/{run_id}", tags=["Pipeline"])
def get_results(run_id: str):
    """
    Retrieve output of a previous pipeline run.
    Survives application restarts by checking SQLite first.
    """
    # 1. Check persistent SQLite store
    stored = db.get_run(run_id)
    if stored:
        return stored

    # 2. Check in-memory store
    if run_id in _run_store:
        return _run_store[run_id]

    raise HTTPException(status_code=404, detail=f"Run ID '{run_id}' not found.")


@app.post("/feedback", tags=["Personalisation"])
def submit_feedback(request: FeedbackRequest):
    """
    Record recruiter feedback on a candidate (shortlisted, rejected, hired).
    Used to personalise subsequent candidate matching and agent recommendations.
    """
    action = request.action.lower().strip()
    if action not in {"shortlisted", "rejected", "hired"}:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid action '{request.action}'. Allowed: 'shortlisted', 'rejected', 'hired'.",
        )

    try:
        entry = db.record_feedback(
            candidate_email=request.candidate_email,
            job_title=request.job_title,
            action=action,
            run_id=request.run_id,
            candidate_summary=request.candidate_summary or "",
            candidate_skills=request.candidate_skills or "",
        )
        return {"status": "recorded", "feedback": entry}
    except Exception as exc:
        logger.error("Failed recording feedback: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/runs", tags=["Pipeline"])
def get_runs(
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """List historical pipeline runs with pagination."""
    items, total = db.list_runs(limit=limit, offset=offset)
    return {
        "runs": items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.post("/chat", tags=["Agent"])
def recruiter_chat(request: ChatRequest):
    """
    Interactive Recruiter Chatbot powered by Gemini function-calling.
    Supports tool execution:
      - parse_job_description
      - search_candidates
      - explain_match
      - record_feedback
      - draft_outreach
    Maintains conversational memory per session_id.
    """
    from agents.recruiter_chat import handle_recruiter_chat

    session_id = request.session_id or str(uuid.uuid4())
    result = handle_recruiter_chat(message=request.message, session_id=session_id)
    return result


# ── Dev server ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
