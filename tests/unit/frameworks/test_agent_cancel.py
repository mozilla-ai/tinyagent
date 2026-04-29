"""Unit tests for `AgentCancel`, `AgentRunError`, and `_unwrap_agent_cancel`."""

from typing import Any
from unittest.mock import patch

import pytest

from tinyagent import AgentCancel, AgentConfig, AgentRunError, TinyAgent
from tinyagent.agent import _unwrap_agent_cancel
from tinyagent.callbacks import Callback, Context
from tinyagent.testing.helpers import DEFAULT_SMALL_MODEL_ID, LLM_IMPORT_PATH
from tinyagent.tracing.agent_trace import AgentTrace


class StopAgent(AgentCancel):
    """Test subclass of `AgentCancel`."""


class SpecificStopAgent(StopAgent):
    """Subclass of a subclass to test deep inheritance."""


class StopAgentRaiser(Callback):
    """Callback that raises `StopAgent` before any LLM interaction."""

    def before_agent_invocation(
        self, context: Context, *args: Any, **kwargs: Any
    ) -> Context:
        msg = "Stopped before LLM"
        raise StopAgent(msg)


class RuntimeErrorRaiser(Callback):
    """Callback that raises a regular `RuntimeError` (should be wrapped in `AgentRunError`)."""

    def before_agent_invocation(
        self, context: Context, *args: Any, **kwargs: Any
    ) -> Context:
        msg = "Unexpected error"
        raise RuntimeError(msg)


def test_agent_cancel_is_abstract() -> None:
    with pytest.raises(TypeError, match="cannot be instantiated directly"):
        AgentCancel("cannot instantiate abstract class")


def test_subclass_can_be_instantiated() -> None:
    exc = StopAgent("test message")
    assert isinstance(exc, AgentCancel)
    assert isinstance(exc, Exception)


def test_subclass_has_trace_property() -> None:
    exc = StopAgent("test")
    assert exc.trace is None


def test_subclass_preserves_message() -> None:
    exc = StopAgent("custom message")
    assert str(exc) == "custom message"


def test_subclass_is_catchable_as_agent_cancel() -> None:
    msg = "test"
    with pytest.raises(AgentCancel):
        raise StopAgent(msg)


def test_subclass_is_catchable_as_specific_type() -> None:
    msg = "test"
    with pytest.raises(StopAgent, match="test"):
        raise StopAgent(msg)


def test_trace_can_be_assigned() -> None:
    exc = StopAgent("test")
    trace = AgentTrace()
    exc._trace = trace
    assert exc.trace is trace


def test_deep_inheritance_works() -> None:
    exc = SpecificStopAgent("deep")
    assert isinstance(exc, AgentCancel)
    assert isinstance(exc, StopAgent)

    msg = "test"
    with pytest.raises(AgentCancel):
        raise SpecificStopAgent(msg)


def test_multiple_args_preserved() -> None:
    exc = StopAgent("arg1", "arg2")
    assert exc.args == ("arg1", "arg2")


@pytest.mark.asyncio
async def test_agent_cancel_propagates_without_wrapping() -> None:
    """`AgentCancel` raised in a callback propagates directly, not wrapped in `AgentRunError`."""
    agent_config = AgentConfig(
        model_id=DEFAULT_SMALL_MODEL_ID,
        callbacks=[StopAgentRaiser()],
    )
    agent = await TinyAgent.create_async(agent_config)

    with patch(LLM_IMPORT_PATH), pytest.raises(StopAgent) as exc_info:
        await agent.run_async("test prompt")

    assert str(exc_info.value) == "Stopped before LLM"
    assert exc_info.value.trace is not None
    assert len(exc_info.value.trace.spans) > 0


@pytest.mark.asyncio
async def test_regular_exception_wrapped_in_agent_run_error() -> None:
    """Regular exceptions are wrapped in `AgentRunError`."""
    agent_config = AgentConfig(
        model_id=DEFAULT_SMALL_MODEL_ID,
        callbacks=[RuntimeErrorRaiser()],
    )
    agent = await TinyAgent.create_async(agent_config)

    with patch(LLM_IMPORT_PATH), pytest.raises(AgentRunError) as exc_info:
        await agent.run_async("test prompt")

    assert isinstance(exc_info.value.original_exception, RuntimeError)
    assert str(exc_info.value.original_exception) == "Unexpected error"
    assert exc_info.value.trace is not None
    assert len(exc_info.value.trace.spans) > 0


def test_unwrap_returns_none_for_regular_exception() -> None:
    exc = RuntimeError("regular error")
    assert _unwrap_agent_cancel(exc) is None


def test_unwrap_returns_none_for_chained_regular_exceptions() -> None:
    inner = ValueError("inner")
    outer = RuntimeError("outer")
    outer.__cause__ = inner
    assert _unwrap_agent_cancel(outer) is None


def test_unwrap_finds_direct_agent_cancel() -> None:
    exc = StopAgent("direct")
    assert _unwrap_agent_cancel(exc) is exc


def test_unwrap_finds_agent_cancel_via_cause() -> None:
    cancel = StopAgent("wrapped")
    wrapper = RuntimeError("framework error")
    wrapper.__cause__ = cancel
    assert _unwrap_agent_cancel(wrapper) is cancel


def test_unwrap_finds_agent_cancel_via_context() -> None:
    cancel = StopAgent("wrapped")
    wrapper = RuntimeError("framework error")
    wrapper.__context__ = cancel
    assert _unwrap_agent_cancel(wrapper) is cancel


def test_unwrap_finds_deeply_nested_agent_cancel() -> None:
    cancel = StopAgent("deep")
    middle = ValueError("middle")
    middle.__cause__ = cancel
    outer = RuntimeError("outer")
    outer.__cause__ = middle
    assert _unwrap_agent_cancel(outer) is cancel


def test_unwrap_prefers_cause_over_context() -> None:
    cause_cancel = StopAgent("from cause")
    context_cancel = SpecificStopAgent("from context")
    wrapper = RuntimeError("wrapper")
    wrapper.__cause__ = cause_cancel
    wrapper.__context__ = context_cancel
    assert _unwrap_agent_cancel(wrapper) is cause_cancel


def test_unwrap_finds_subclass_of_agent_cancel() -> None:
    cancel = SpecificStopAgent("specific")
    wrapper = RuntimeError("wrapper")
    wrapper.__cause__ = cancel
    result = _unwrap_agent_cancel(wrapper)
    assert result is cancel
    assert isinstance(result, StopAgent)
    assert isinstance(result, AgentCancel)
