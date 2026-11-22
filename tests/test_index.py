"""
tests/test_index.py
--------------------
Unit tests for CandidateIndex:
- Index creation and profile text generation
- Cosine similarity search ordering
- Incremental add & remove
- On-disk save & load round-trip
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from retrieval.embedder import FakeEmbedder
from retrieval.index import CandidateIndex, build_candidate_profile_text


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "name": "Alice Sharma",
            "email": "alice@example.com",
            "skills": "Python, machine learning, scikit-learn, NLP, FastAPI",
            "years_experience": 5,
            "location": "San Francisco",
            "summary": "ML engineer with focus on NLP and production model deployment.",
        },
        {
            "name": "Bob Zhang",
            "email": "bob@example.com",
            "skills": "Java, Spring Boot, SQL, REST APIs, Docker, Kubernetes",
            "years_experience": 7,
            "location": "New York",
            "summary": "Backend software engineer specialising in Java microservices.",
        },
        {
            "name": "Priya Mehta",
            "email": "priya@example.com",
            "skills": "Python, data analysis, pandas, NumPy, SQL, Tableau",
            "years_experience": 4,
            "location": "Austin",
            "summary": "Data analyst with strong Python and Tableau skills.",
        },
    ])


def test_profile_text_excludes_name(sample_df):
    row = sample_df.iloc[0]
    text = build_candidate_profile_text(row)
    assert "Alice Sharma" not in text
    assert "Python" in text
    assert "5 years" in text


def test_index_build_and_search(sample_df):
    embedder = FakeEmbedder(dimension=32)
    index = CandidateIndex(embedder=embedder)
    index.build_from_dataframe(sample_df)

    assert len(index) == 3
    assert index.vectors.shape == (3, 32)

    # Search with dummy query vector
    q_vec = embedder.embed_texts(["Python machine learning FastAPI"])[0]
    results = index.search(q_vec, top_k=2)

    assert len(results) == 2
    assert results[0][1] >= results[1][1]  # Sorted descending
    assert "email" in results[0][0]


def test_index_incremental_add_and_remove(sample_df):
    embedder = FakeEmbedder(dimension=32)
    index = CandidateIndex(embedder=embedder)
    index.build_from_dataframe(sample_df)

    new_cand = {
        "name": "Devin Miller",
        "email": "devin@example.com",
        "skills": "Rust, systems programming",
        "years_experience": 3,
        "location": "Remote",
        "summary": "Systems engineer writing low latency code.",
    }

    # Add candidate
    index.add(new_cand)
    assert len(index) == 4
    assert index.vectors.shape == (4, 32)
    assert "devin@example.com" in index.df["email"].values

    # Remove candidate
    removed = index.remove("devin@example.com")
    assert removed is True
    assert len(index) == 3
    assert index.vectors.shape == (3, 32)
    assert "devin@example.com" not in index.df["email"].values

    # Remove non-existent candidate
    assert index.remove("ghost@example.com") is False


def test_index_save_and_load_round_trip(sample_df):
    temp_dir = Path(tempfile.mkdtemp())
    try:
        embedder = FakeEmbedder(dimension=32)
        index = CandidateIndex(embedder=embedder)
        index.build_from_dataframe(sample_df)

        index.save(temp_dir)
        loaded = CandidateIndex.load(temp_dir, embedder=embedder)

        assert len(loaded) == len(index)
        assert loaded.vectors.shape == index.vectors.shape
        assert np.allclose(loaded.vectors, index.vectors)
        assert list(loaded.df["email"]) == list(index.df["email"])
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
