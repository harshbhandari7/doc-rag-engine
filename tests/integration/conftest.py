"""Fixtures for integration tests.

Integration tests require Qdrant running and populated collections.
Skip them in CI or when Docker is not available:

    pytest tests/unit/             # fast, no external deps
    pytest tests/ -m integration   # integration tests only
    pytest tests/                  # everything
"""
from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration tests unless -m integration is explicitly passed."""
    if config.getoption("-m", default="") == "integration":
        return
    skip = pytest.mark.skip(reason="integration tests require Qdrant — run with -m integration")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def cfg():
    from configs.settings import AppConfig
    return AppConfig()


@pytest.fixture(scope="session")
def embedder():
    from src.embedding.embedder import Embedder
    return Embedder()


@pytest.fixture(scope="session")
def store():
    from src.vectorstore.store import VectorStore
    return VectorStore()
