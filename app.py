"""
app.py
------
FastAPI entry point for the HireFlow multi-agent recruitment pipeline.

Endpoints:
  GET  /health            — health check
  POST /run               — run the full 4-agent pipeline
  GET  /results/{run_id}  — retrieve a previous run's output (in-memory store)
"""

import base64
import io
import logging
import os
import uuid
from datetime import datetime
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from pipeline.graph import pipeline

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HireFlow",
    description="Multi-agent recruitment pipeline: JD Parsing → Candidate Matching → Outreach → Scheduling",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory run store ────────────────────────────────────────────────────────
# In production, replace with a database or Redis.
_run_store: dict[str, dict] = {}

# ── Request / Response models ──────────────────────────────────────────────────

class RunRequest(BaseModel):
    job_description: str
    candidates_csv: Optional[str] = None        # base64-encoded CSV content OR a file path
    candidates_csv_path: Optional[str] = None   # direct filesystem path (for local dev)


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


# ── Helper ─────────────────────────────────────────────────────────────────────

def _resolve_csv_path(request: RunRequest) -> str:
    """
    Determine the CSV path to use.
    Priority: direct path > base64 content (written to a temp file) > default sample
    """
    if request.candidates_csv_path:
        return request.candidates_csv_path

    if request.candidates_csv:
        try:
            csv_bytes = base64.b64decode(request.candidates_csv)
            tmp_path = f"data/uploaded_{uuid.uuid4().hex[:8]}.csv"
            os.makedirs("data", exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(csv_bytes)
            logger.info("Wrote uploaded CSV to %s", tmp_path)
            return tmp_path
        except Exception as exc:
            logger.warning("Could not decode base64 CSV — using sample. Error: %s", exc)

    return "data/sample_candidates.csv"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns a simple health status for uptime monitoring."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="2.0.0",
    )


@app.post("/run", tags=["Pipeline"])
def run_pipeline(request: RunRequest):
    """
    Run the full HireFlow multi-agent pipeline.

    - **job_description**: Raw JD text (required)
    - **candidates_csv**: Base64-encoded CSV (optional; uses sample data if omitted)
    - **candidates_csv_path**: Direct filesystem path to a CSV (optional; for local dev)

    Returns a run_id plus the complete structured output from all 4 agents.
    """
    if not request.job_description.strip():
        raise HTTPException(status_code=422, detail="job_description cannot be empty.")

    csv_path = _resolve_csv_path(request)
    run_id = str(uuid.uuid4())

    initial_state = {
        "raw_jd": request.job_description,
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

    _run_store[run_id] = result
    logger.info("Pipeline run %s completed. Errors: %s", run_id, result["errors"])
    return result


@app.get("/results/{run_id}", tags=["Pipeline"])
def get_results(run_id: str):
    """Retrieve the output of a previous pipeline run by its run_id."""
    if run_id not in _run_store:
        raise HTTPException(status_code=404, detail=f"Run ID '{run_id}' not found.")
    return _run_store[run_id]


# ── Dev server ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
