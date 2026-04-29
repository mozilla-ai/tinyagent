import os
from collections.abc import Generator
from unittest.mock import patch

import pytest
from any_llm import AnyLLM


@pytest.fixture(autouse=True)
def mock_verify_api_key() -> Generator[None, None, None]:
    """Mock AnyLLM._verify_and_set_api_key to skip API key validation in unit tests."""
    with patch("any_llm.AnyLLM._verify_and_set_api_key"):
        yield


@pytest.fixture(autouse=True)
def clear_any_llm_key() -> Generator[None, None, None]:
    """Clear ANY_LLM_KEY to prevent platform provider path in unit tests."""
    key = getattr(AnyLLM, "ANY_LLM_KEY", None)
    original = os.environ.pop(key, None) if key else None
    yield
    if original is not None and key is not None:
        os.environ[key] = original


@pytest.fixture(autouse=True)
def mock_api_keys_for_unit_tests() -> Generator[None, None, None]:
    """Provide a dummy Mistral API key so the default model can be initialized in tests."""
    os.environ.setdefault("MISTRAL_API_KEY", "dummy-mistral-key-for-unit-tests")
    yield  # noqa: PT022
