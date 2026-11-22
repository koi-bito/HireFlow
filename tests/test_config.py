"""
tests/test_config.py
---------------------
Tests for configuration loading and defaults.
"""

import os
from unittest.mock import patch

import config


def test_default_config_values():
    assert config.MATCHER_MODE in {"keyword", "embedding", "hybrid"}
    assert config.RETRIEVAL_TOP_K > 0
    assert config.RERANK_TOP_N > 0
    assert config.EMBEDDING_MODEL
    assert config.EMBEDDING_CACHE_DIR is not None
    assert isinstance(config.CORS_ORIGINS, list)


def test_custom_env_override():
    with patch.dict(os.environ, {"MATCHER_MODE": "keyword", "RETRIEVAL_TOP_K": "25"}):
        import importlib
        importlib.reload(config)
        assert config.MATCHER_MODE == "keyword"
        assert config.RETRIEVAL_TOP_K == 25
    # Reload again to restore default
    importlib.reload(config)
