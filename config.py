"""
config.py
---------
Configuration management for HireFlow v3.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent

# LLM & API settings
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-3.5-flash")
GOOGLE_GENAI_USE_VERTEXAI = os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE").upper() == "TRUE"

# Matcher & Retrieval Configuration
# Modes: "keyword", "embedding", "hybrid"
MATCHER_MODE = os.getenv("MATCHER_MODE", "hybrid").lower()

# Embedding model: Gemini embedding model or sentence-transformers
# Default Gemini embedding model via google-genai
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LOCAL_FALLBACK_EMBEDDING_MODEL = os.getenv("LOCAL_FALLBACK_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Retrieval & Ranking knobs
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "10"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))

# Cache directory for embedded vectors
EMBEDDING_CACHE_DIR = Path(os.getenv("EMBEDDING_CACHE_DIR", str(BASE_DIR / ".cache" / "embeddings")))

# Scoring weights and boosts
WEIGHT_RETRIEVAL = float(os.getenv("WEIGHT_RETRIEVAL", "0.4"))
WEIGHT_LLM = float(os.getenv("WEIGHT_LLM", "0.6"))
EXPERIENCE_BOOST_WEIGHT = float(os.getenv("EXPERIENCE_BOOST_WEIGHT", "0.05"))
LOCATION_BOOST_WEIGHT = float(os.getenv("LOCATION_BOOST_WEIGHT", "0.05"))
RRF_K = int(os.getenv("RRF_K", "60"))

# Feedback & Persistence
DB_PATH = Path(os.getenv("DB_PATH", str(BASE_DIR / "hireflow.db")))

# Outreach & Scheduling limits
OUTREACH_MAX_CANDIDATES = int(os.getenv("OUTREACH_MAX_CANDIDATES", "5"))

# Security / CORS
CORS_ORIGINS = [orig.strip() for orig in os.getenv("CORS_ORIGINS", "*").split(",") if orig.strip()]
