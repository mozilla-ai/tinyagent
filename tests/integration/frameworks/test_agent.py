"""Integration smoke tests for the TinyAgent loop.

These tests require a real `MISTRAL_API_KEY` (or another provider) and a working
network connection. They are skipped automatically when credentials are missing.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import timedelta

import pytest

from tinyagent import AgentConfig, TinyAgent
from tinyagent.config import MCPStdio
from tinyagent.testing.helpers import (
    DEFAULT_SMALL_MODEL_ID,
    get_default_agent_model_args,
    group_spans,
)
from tinyagent.tracing.agent_trace import AgentSpan, AgentTrace
from tinyagent.tracing.attributes import GenAI


def _uvx_installed() -> bool:
    try:
        subprocess.run(  # noqa: S603
            ["uvx", "--version"],  # noqa: S607
            capture_output=True,
            check=True,
        )
    except Exception:
        return False
    return True


def _has_mistral_key() -> bool:
    return bool(os.environ.get("MISTRAL_API_KEY"))


pytestmark = [
    pytest.mark.skipif(not _has_mistral_key(), reason="MISTRAL_API_KEY not set"),
    pytest.mark.skipif(not _uvx_installed(), reason="uvx not installed"),
]


def _assert_first_llm_call(llm_call: AgentSpan) -> None:
    input_messages_raw = llm_call.attributes.get(GenAI.INPUT_MESSAGES)
    assert input_messages_raw is not None
    input_messages = json.loads(input_messages_raw)
    assert input_messages[0]["role"] == "system"
    assert input_messages[1]["role"] == "user"


def _assert_first_tool_execution(tool_execution: AgentSpan) -> None:
    tool_args_raw = tool_execution.attributes.get(GenAI.TOOL_ARGS)
    assert tool_args_raw is not None
    args = json.loads(tool_args_raw)
    assert "timezone" in args


def test_tinyagent_with_mcp_time_server() -> None:
    """Run TinyAgent against the time MCP server and verify the trace shape."""
    config = AgentConfig(
        model_id=DEFAULT_SMALL_MODEL_ID,
        instructions="Use the available tools to answer the user's question.",
        tools=[MCPStdio(command="uvx", args=["mcp-server-time"])],
        model_args=get_default_agent_model_args(DEFAULT_SMALL_MODEL_ID),
    )

    agent = TinyAgent.create(config)
    try:
        trace = agent.run("What time is it in UTC right now?")
    finally:
        # Sync drain of MCP clients
        import asyncio

        asyncio.run(agent.cleanup_async())

    assert isinstance(trace, AgentTrace)
    assert trace.final_output

    invocations, llm_calls, tool_executions = group_spans(trace.spans)
    assert len(invocations) == 1
    assert len(llm_calls) >= 1
    _assert_first_llm_call(llm_calls[0])
    if tool_executions:
        _assert_first_tool_execution(tool_executions[0])

    assert isinstance(trace.duration, timedelta)
    assert trace.duration.total_seconds() > 0
