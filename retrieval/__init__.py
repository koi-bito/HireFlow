"""
retrieval package
-----------------
Vector embeddings, candidate indexing, and similarity retrieval for HireFlow v3.
"""

from retrieval.embedder import Embedder, FakeEmbedder, get_embedder
from retrieval.index import CandidateIndex

__all__ = ["Embedder", "FakeEmbedder", "get_embedder", "CandidateIndex"]
