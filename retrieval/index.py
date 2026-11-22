"""
retrieval/index.py
------------------
In-memory vector index for candidate profiles with incremental updates and disk persistence.
Uses vectorized NumPy matrix multiplication for cosine similarity search.
"""

import json
import logging
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd

from retrieval.embedder import Embedder

logger = logging.getLogger(__name__)


def build_candidate_profile_text(row: Union[pd.Series, dict[str, Any]]) -> str:
    """
    Constructs search profile text excluding candidate name to avoid demographic bias.
    Uses skills, years of experience, location, and professional summary.
    """
    skills = str(row.get("skills", "")).strip()
    exp = str(row.get("years_experience", "")).strip()
    loc = str(row.get("location", "")).strip()
    summary = str(row.get("summary", "")).strip()

    parts = []
    if skills:
        parts.append(f"Skills: {skills}")
    if exp:
        parts.append(f"Experience: {exp} years")
    if loc:
        parts.append(f"Location: {loc}")
    if summary:
        parts.append(f"Summary: {summary}")

    return " | ".join(parts) if parts else "No profile data provided"


class CandidateIndex:
    """
    Vector search index for candidate profiles.
    Backed by a pandas DataFrame for metadata and an (N, D) L2-normalized float32 NumPy array for vectors.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.df: pd.DataFrame = pd.DataFrame()
        self.vectors: np.ndarray = np.empty((0, 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.df)

    def build_from_dataframe(self, df: pd.DataFrame) -> "CandidateIndex":
        """Build or re-build the entire index from a candidate DataFrame."""
        if df.empty:
            self.df = pd.DataFrame()
            self.vectors = np.empty((0, 0), dtype=np.float32)
            return self

        # Keep original columns, ensure string fills
        clean_df = df.copy().fillna("")
        if "email" in clean_df.columns:
            clean_df = clean_df.drop_duplicates(subset=["email"], keep="first")
        clean_df = clean_df.reset_index(drop=True)

        profile_texts = [build_candidate_profile_text(row) for _, row in clean_df.iterrows()]
        vectors = self.embedder.embed_texts(profile_texts)

        self.df = clean_df
        self.vectors = vectors
        logger.info("CandidateIndex built with %d candidates (dim=%d)", len(self.df), self.vectors.shape[1] if len(self.vectors) else 0)
        return self

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> list[tuple[dict[str, Any], float]]:
        """
        Cosine similarity search against all candidates in the index.
        Returns list of (candidate_dict, score) sorted descending by score.
        """
        if len(self.df) == 0 or self.vectors.size == 0:
            return []

        # Ensure query vector is (D,) and normalized
        q = query_vector.flatten()
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        # Dot product with normalized candidate vectors = cosine similarity
        scores = np.dot(self.vectors, q)

        # Clip scores to [0.0, 1.0] range
        scores = np.clip(scores, 0.0, 1.0)

        # Sort indices
        k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            row_dict = self.df.iloc[idx].to_dict()
            results.append((row_dict, float(scores[idx])))

        return results

    def add(self, candidates: Union[dict[str, Any], list[dict[str, Any]], pd.DataFrame]) -> None:
        """
        Incrementally add one or more candidate profiles to the index.
        If a candidate's email already exists, updates their row and vector.
        """
        if isinstance(candidates, dict):
            new_df = pd.DataFrame([candidates])
        elif isinstance(candidates, list):
            new_df = pd.DataFrame(candidates)
        else:
            new_df = candidates.copy()

        new_df = new_df.fillna("").reset_index(drop=True)
        if new_df.empty:
            return

        for _, row in new_df.iterrows():
            email = row.get("email")
            # If email exists, remove old entry first
            if email and not self.df.empty and (self.df["email"] == email).any():
                self.remove(email)

            # Embed single candidate
            text = build_candidate_profile_text(row)
            new_vec = self.embedder.embed_texts([text])

            if self.df.empty:
                self.df = pd.DataFrame([row.to_dict()])
                self.vectors = new_vec
            else:
                self.df = pd.concat([self.df, pd.DataFrame([row.to_dict()])], ignore_index=True)
                self.vectors = np.vstack([self.vectors, new_vec])

        logger.debug("Added %d candidates to index (total=%d)", len(new_df), len(self.df))

    def remove(self, email: str) -> bool:
        """
        Incrementally remove a candidate by email. Returns True if removed, False if not found.
        """
        if self.df.empty or "email" not in self.df.columns:
            return False

        mask = (self.df["email"] == email).to_numpy()
        if not mask.any():
            return False

        idx_to_remove = np.where(mask)[0][0]
        self.df = self.df.drop(index=idx_to_remove).reset_index(drop=True)
        self.vectors = np.delete(self.vectors, idx_to_remove, axis=0)
        logger.debug("Removed candidate %s from index (total=%d)", email, len(self.df))
        return True

    def save(self, directory: Union[str, Path]) -> None:
        """Persist index dataframe and vectors to disk."""
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Save dataframe as CSV
        csv_path = target_dir / "candidates.csv"
        self.df.to_csv(csv_path, index=False)

        # Save vectors as npy
        vec_path = target_dir / "vectors.npy"
        np.save(vec_path, self.vectors)

        # Save metadata
        meta_path = target_dir / "metadata.json"
        meta = {
            "count": len(self.df),
            "dim": int(self.vectors.shape[1]) if self.vectors.size > 0 else 0,
            "embedder_model": self.embedder.model_name,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Saved CandidateIndex to %s", target_dir)

    @classmethod
    def load(cls, directory: Union[str, Path], embedder: Embedder) -> "CandidateIndex":
        """Load persisted index from disk."""
        target_dir = Path(directory)
        csv_path = target_dir / "candidates.csv"
        vec_path = target_dir / "vectors.npy"

        if not csv_path.exists() or not vec_path.exists():
            raise FileNotFoundError(f"Missing index files in {target_dir}")

        index = cls(embedder=embedder)
        index.df = pd.read_csv(csv_path).fillna("")
        index.vectors = np.load(vec_path).astype(np.float32)

        logger.info("Loaded CandidateIndex from %s (%d candidates)", target_dir, len(index.df))
        return index
