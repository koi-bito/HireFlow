"""
retrieval/embedder.py
---------------------
Embedding abstraction layer with Gemini, Sentence-Transformers, and Fake (hash-based) implementations.
Includes disk-based caching and exponential backoff retry.
"""

import abc
import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

import config

logger = logging.getLogger(__name__)


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a 2D float numpy array. Safe against zero-norm vectors."""
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (vectors / norms).astype(np.float32)


class Embedder(abc.ABC):
    """Abstract Base Class for text embedding models."""

    def __init__(self, cache_dir: Optional[Path] = None, model_name: str = "generic"):
        self.model_name = model_name
        self.cache_dir = cache_dir or config.EMBEDDING_CACHE_DIR
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, text: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        text_hash = hashlib.sha256(f"{self.model_name}::{text}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{text_hash}.npy"

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of strings into an (N, D) L2-normalized numpy array.
        Uses on-disk cache if available.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        results: list[Optional[np.ndarray]] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        # Check cache
        for idx, text in enumerate(texts):
            cache_path = self._get_cache_path(text)
            if cache_path and cache_path.exists():
                try:
                    vec = np.load(cache_path)
                    results[idx] = vec
                    continue
                except Exception as exc:
                    logger.debug("Failed reading cache file %s: %s", cache_path, exc)

            missing_indices.append(idx)
            missing_texts.append(text)

        # Compute missing embeddings
        if missing_texts:
            new_vectors = self._embed_batch(missing_texts)
            new_vectors = normalize_vectors(new_vectors)

            for idx, vec, text in zip(missing_indices, new_vectors, missing_texts):
                results[idx] = vec
                cache_path = self._get_cache_path(text)
                if cache_path:
                    try:
                        np.save(cache_path, vec)
                    except Exception as exc:
                        logger.debug("Failed writing cache file %s: %s", cache_path, exc)

        return np.vstack(results).astype(np.float32)

    @abc.abstractmethod
    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Subclasses must implement actual batch embedding."""
        pass


class FakeEmbedder(Embedder):
    """
    Deterministic feature-hashing embedder for fast offline testing and CI.
    Uses token and subword hashing to project text into a normalized vector space.
    Shared terms naturally produce positive cosine similarity, enabling accurate offline ranking evals.
    """

    def __init__(self, dimension: int = 512, cache_dir: Optional[Path] = None):
        super().__init__(cache_dir=cache_dir, model_name="fake-embedder")
        self.dimension = dimension
        self.call_count = 0  # Useful for verifying cache hits in tests

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        import re

        self.call_count += 1
        vectors = []
        for text in texts:
            vec = np.zeros(self.dimension, dtype=np.float32)
            tokens = re.findall(r"[a-zA-Z0-9_\+\#\.]+", text.lower())
            if not tokens:
                vectors.append(vec)
                continue

            for token in tokens:
                h = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(h[:4], byteorder="big") % self.dimension
                vec[idx] += 1.0

            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)

        return np.vstack(vectors)


class GeminiEmbedder(Embedder):
    """
    Embedder using Google GenAI SDK (google-genai client).
    Batches requests and retries with backoff on rate limits / network errors.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        batch_size: int = 32,
    ):
        model = model_name or config.EMBEDDING_MODEL
        super().__init__(cache_dir=cache_dir, model_name=model)
        self.api_key = api_key or config.GOOGLE_API_KEY
        self.batch_size = batch_size
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        all_vecs = []

        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            max_retries = 3
            backoff = 1.0

            for attempt in range(max_retries):
                try:
                    response = client.models.embed_content(
                        model=self.model_name,
                        contents=chunk,
                    )
                    # Extract embeddings
                    chunk_vecs = []
                    for emb in response.embeddings:
                        chunk_vecs.append(np.array(emb.values, dtype=np.float32))
                    all_vecs.extend(chunk_vecs)
                    break
                except Exception as exc:
                    if attempt == max_retries - 1:
                        logger.error("Gemini embedding failed after %d attempts: %s", max_retries, exc)
                        raise
                    logger.warning("Gemini embedding transient error (attempt %d/%d): %s. Retrying in %.1fs", attempt + 1, max_retries, exc, backoff)
                    time.sleep(backoff)
                    backoff *= 2.0

        return np.vstack(all_vecs)


class SentenceTransformerEmbedder(Embedder):
    """
    Offline local embedder using sentence-transformers (e.g. all-MiniLM-L6-v2).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        batch_size: int = 32,
    ):
        model = model_name or config.LOCAL_FALLBACK_EMBEDDING_MODEL
        super().__init__(cache_dir=cache_dir, model_name=model)
        self.batch_size = batch_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError as err:
                raise ImportError(
                    "sentence-transformers is not installed. Run `pip install sentence-transformers`."
                ) from err
        return self._model

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        model = self._get_model()
        vecs = model.encode(texts, batch_size=self.batch_size, normalize_embeddings=False)
        return np.array(vecs, dtype=np.float32)


def get_embedder(
    mode: str = "default",
    cache_dir: Optional[Path] = None,
    dimension: int = 64,
) -> Embedder:
    """
    Factory function to instantiate the configured Embedder.
    - 'fake' -> FakeEmbedder
    - 'gemini' -> GeminiEmbedder
    - 'sentence-transformers' / 'local' -> SentenceTransformerEmbedder
    - 'default' -> GeminiEmbedder if GOOGLE_API_KEY else FakeEmbedder
    """
    if mode == "fake":
        return FakeEmbedder(dimension=dimension, cache_dir=cache_dir)
    elif mode in {"sentence-transformers", "local"}:
        return SentenceTransformerEmbedder(cache_dir=cache_dir)
    elif mode == "gemini":
        return GeminiEmbedder(cache_dir=cache_dir)

    # default: prefer FakeEmbedder in test mode or if offline flag is set
    if os.getenv("HIREFLOW_TEST_MODE", "").upper() == "1" or os.getenv("HIREFLOW_OFFLINE", "").upper() == "TRUE":
        return FakeEmbedder(dimension=dimension, cache_dir=cache_dir)

    if config.GOOGLE_API_KEY:
        return GeminiEmbedder(cache_dir=cache_dir)
    return FakeEmbedder(dimension=dimension, cache_dir=cache_dir)
