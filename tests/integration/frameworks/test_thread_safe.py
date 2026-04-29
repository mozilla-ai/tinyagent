"""Concurrency smoke test: state from one run should not bleed into another."""

import asyncio
import os

import pytest

from tinyagent import AgentConfig, TinyAgent
from tinyagent.testing.helpers import (
    DEFAULT_SMALL_MODEL_ID,
    get_default_agent_model_args,
)


pytestmark = pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"), reason="MISTRAL_API_KEY not set"
)


@pytest.mark.asyncio
async def test_run_agent_concurrently() -> None:
    """When an agent is run concurrently, state from the first run shouldn't bleed into the second run."""

    def mock_capital(query: str) -> str:
        """Look up the capital of a country.

        Args:
            query: The country to look up.

        Returns:
            A short fact about the capital.

        """
        if "France" in query:
            return "The capital of France is Paris."
        if "Spain" in query:
            return "The capital of Spain is Madrid."
        return "No info"

    agent = await TinyAgent.create_async(
        AgentConfig(
            model_id=DEFAULT_SMALL_MODEL_ID,
            instructions="You must use the tools to find an answer",
            model_args=get_default_agent_model_args(),
            tools=[mock_capital],
        )
    )

    results = await asyncio.gather(
        agent.run_async("What is the capital of France?"),
        agent.run_async("What is the capital of Spain?"),
    )
    outputs = [r.final_output for r in results]
    assert all(o is not None for o in outputs)

    assert sum("Paris" in str(o) for o in outputs) == 1
    assert sum("Madrid" in str(o) for o in outputs) == 1

    first_spans = results[0].spans
    second_spans = results[1].spans
    assert second_spans[: len(first_spans)] != first_spans, (
        "Spans from the first run should not appear in the second"
    )
