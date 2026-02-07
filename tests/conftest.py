from __future__ import annotations

import pytest

from usc.bench.datasets import toy_agent_log, toy_big_agent_log, toy_big_agent_log_varied


@pytest.fixture
def toy_text() -> str:
    return toy_agent_log()


@pytest.fixture
def big_text() -> str:
    return toy_big_agent_log()


@pytest.fixture
def varied_text() -> str:
    return toy_big_agent_log_varied()


@pytest.fixture
def tmp_store(tmp_path):
    return tmp_path / "commits.jsonl"


# ---------- optional-dep skip markers ----------

def _can_import(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


requires_numpy = pytest.mark.skipif(
    not _can_import("numpy"), reason="numpy not installed"
)
requires_sklearn = pytest.mark.skipif(
    not _can_import("sklearn"), reason="scikit-learn not installed"
)
requires_cryptography = pytest.mark.skipif(
    not _can_import("cryptography"), reason="cryptography not installed"
)
requires_otel = pytest.mark.skipif(
    not _can_import("opentelemetry"), reason="opentelemetry not installed"
)
requires_langchain = pytest.mark.skipif(
    not _can_import("langchain_core"), reason="langchain-core not installed"
)
