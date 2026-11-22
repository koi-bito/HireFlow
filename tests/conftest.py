"""
tests/conftest.py
-----------------
Pytest configuration and global test hooks.
Sets test environment flags to ensure all tests run offline deterministically.
"""

import os

# Ensure tests run fully offline with deterministic fake embedder
os.environ["HIREFLOW_TEST_MODE"] = "1"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
