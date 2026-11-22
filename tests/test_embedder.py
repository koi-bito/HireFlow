"""
tests/test_embedder.py
-----------------------
Unit tests for the Embedder layer:
- FakeEmbedder determinism and dimension
- Normalization (unit length)
- On-disk caching and cache hits
- Batching handling
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

from retrieval.embedder import FakeEmbedder, normalize_vectors


def test_normalize_vectors():
    vecs = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normed = normalize_vectors(vecs)
    # First row norm should be 1.0 (3^2+4^2=25, sqrt=5 -> [0.6, 0.8])
    assert np.isclose(np.linalg.norm(normed[0]), 1.0)
    # Zero vector should not divide by zero or produce NaN
    assert not np.isnan(normed[1]).any()


def test_fake_embedder_determinism():
    embedder = FakeEmbedder(dimension=64)
    v1 = embedder.embed_texts(["Python NLP Engineer"])
    v2 = embedder.embed_texts(["Python NLP Engineer"])
    v3 = embedder.embed_texts(["Java Spring Developer"])

    assert v1.shape == (1, 64)
    assert np.allclose(v1, v2)
    assert not np.allclose(v1, v3)
    assert np.isclose(np.linalg.norm(v1[0]), 1.0)


def test_embedder_cache_hit():
    temp_dir = Path(tempfile.mkdtemp())
    try:
        embedder = FakeEmbedder(dimension=32, cache_dir=temp_dir)
        text = "Senior Distributed Systems Engineer"

        # First call: cache miss, call_count increments
        v1 = embedder.embed_texts([text])
        assert embedder.call_count == 1

        # Second call: cache hit, call_count does NOT increment
        v2 = embedder.embed_texts([text])
        assert embedder.call_count == 1
        assert np.allclose(v1, v2)

        # Distinct text: cache miss, call_count increments
        embedder.embed_texts(["Product Manager"])
        assert embedder.call_count == 2
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_embedder_empty_and_batch_inputs():
    embedder = FakeEmbedder(dimension=16)
    empty = embedder.embed_texts([])
    assert empty.shape == (0, 0)

    texts = [f"Candidate profile {i}" for i in range(25)]
    vecs = embedder.embed_texts(texts)
    assert vecs.shape == (25, 16)
    for v in vecs:
        assert np.isclose(np.linalg.norm(v), 1.0)
