"""Error-handling integration tests for the TinyAgent loop."""

import os
from typing import Any
from unittest.mock import patch

import pytest

from tinyagent import (
    AgentCancel,
    AgentConfig,
    AgentRunError,
    TinyAgent,
)
from tinyagent.callbacks import Callback, Context
from tinyagent.testing.helpers import (
    DEFAULT_SMALL_MODEL_ID,
    LLM_IMPORT_PATH,
    get_default_agent_model_args,
)
from tinyagent.tracing.otel_types import StatusCode


pytestmark = pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"), reason="MISTRAL_API_KEY not set"
)


class LimitLLMCalls(Callback):
    def __init__(self, max_llm_calls: int) -> None:
        self.max_llm_calls = max_llm_calls

    def before_llm_call(self, context: Context, *args: Any, **kwargs: Any) -> Context:
        if "n_llm_calls" not in context.shared:
            context.shared["n_llm_calls"] = 0

        context.shared["n_llm_calls"] += 1

        if context.shared["n_llm_calls"] > self.max_llm_calls:
            msg = "Reached limit of LLM Calls"
            raise RuntimeError(msg)

        return context


def test_runtime_error() -> None:
    """An exception thrown during the LLM call is caught and surfaced via `AgentRunError.trace`."""
    test_runtime_error_msg = "runtime error trap"

    with patch(LLM_IMPORT_PATH) as llm_completion_path:
        llm_completion_path.side_effect = RuntimeError(test_runtime_error_msg)
        agent_config = AgentConfig(
            model_id=DEFAULT_SMALL_MODEL_ID,
            tools=[],
            model_args=get_default_agent_model_args(),
        )
        agent = TinyAgent.create(agent_config)
        spans = []
        try:
            agent.run("Write a four-line poem about agent frameworks.")
        except AgentRunError as are:
            spans = are.trace.spans
            assert any(
                span.status.status_code == StatusCode.ERROR
                and span.status.description is not None
                and test_runtime_error_msg in span.status.description
                for span in spans
            )


def test_tool_error() -> None:
    """A tool that raises is recorded with ERROR status; the agent is allowed to recover."""
    exception_reason = "tool error trap"

    def search_web(query: str) -> str:
        """Perform a duckduckgo web search.

        Args:
            query: The search query to perform.

        Returns:
            The top search results.

        """
        msg = exception_reason
        raise ValueError(msg)

    agent_config = AgentConfig(
        model_id=DEFAULT_SMALL_MODEL_ID,
        instructions="You must use the available tools to answer questions.",
        tools=[search_web],
        model_args=get_default_agent_model_args(),
        callbacks=[LimitLLMCalls(max_llm_calls=5)],
    )

    agent = TinyAgent.create(agent_config)
    agent_trace = agent.run(
        "Check in the web which agent framework is the best. "
        "If the tool fails, don't try again, return final answer as failure.",
    )
    assert any(
        span.is_tool_execution()
        and span.status.status_code == StatusCode.ERROR
        and exception_reason in getattr(span.status, "description", "")
        for span in agent_trace.spans
    )


class StopExecution(AgentCancel):
    """Test exception for cancelling agent execution."""


class StopBeforeFirstLLMCall(Callback):
    def before_llm_call(self, context: Context, *args: Any, **kwargs: Any) -> Context:
        msg = "Stopped by callback"
        raise StopExecution(msg)


def test_agent_cancel_not_wrapped() -> None:
    """AgentCancel subclasses propagate without being wrapped in AgentRunError, with the trace attached."""
    agent_config = AgentConfig(
        model_id=DEFAULT_SMALL_MODEL_ID,
        tools=[],
        callbacks=[StopBeforeFirstLLMCall()],
        model_args=get_default_agent_model_args(),
    )
    agent = TinyAgent.create(agent_config)

    with pytest.raises(StopExecution) as exc_info:
        agent.run("test")

    assert exc_info.value.trace is not None
    assert len(exc_info.value.trace.spans) > 0
