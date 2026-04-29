import asyncio
import time
from collections.abc import Sequence
from typing import Any

import httpx
import requests

from tinyagent.tracing.agent_trace import AgentSpan

DEFAULT_SMALL_MODEL_ID = "mistral:mistral-small-latest"

LLM_IMPORT_PATH = "any_llm.AnyLLM.acompletion"


def get_default_agent_model_args(model_id: str | None = None) -> dict[str, Any]:
    """Get the default model arguments for tinyagent in tests.

    Args:
        model_id: The model ID to get specific model arguments for.
            Currently unused, kept for forward compatibility.

    Returns:
        The default model arguments.

    """
    del model_id
    model_args: dict[str, Any] = {"parallel_tool_calls": False, "temperature": 0.0}
    return model_args


DEFAULT_HTTP_KWARGS = {"timeout": 60.0}


def wait_for_server(
    server_url: str, max_attempts: int = 20, poll_interval: float = 0.5
) -> None:
    """Wait for a server to be ready.

    Args:
        server_url: The URL of the server to wait for.
        max_attempts: The maximum number of attempts to make.
        poll_interval: The interval between attempts.

    """
    attempts = 0
    while True:
        try:
            requests.get(server_url, timeout=1.0)
            return  # noqa: TRY300
        except (requests.RequestException, ConnectionError):
            pass

        time.sleep(poll_interval)
        attempts += 1
        if attempts >= max_attempts:
            msg = f"Could not connect to {server_url}. Tried {max_attempts} times with {poll_interval} second interval."
            raise ConnectionError(msg)


async def wait_for_server_async(
    server_url: str, max_attempts: int = 20, poll_interval: float = 0.5
) -> None:
    """Wait for a server to be ready (async)."""
    attempts = 0

    async with httpx.AsyncClient() as client:
        while True:
            try:
                await client.get(server_url, timeout=1.0)
                return  # noqa: TRY300
            except (httpx.RequestError, httpx.TimeoutException):
                pass

            await asyncio.sleep(poll_interval)
            attempts += 1
            if attempts >= max_attempts:
                msg = f"Could not connect to {server_url}. Tried {max_attempts} times with {poll_interval} second interval."
                raise ConnectionError(msg)


def group_spans(
    spans: Sequence[AgentSpan],
) -> tuple[Sequence[AgentSpan], Sequence[AgentSpan], Sequence[AgentSpan]]:
    """Group spans into agent invocations, llm calls and tool executions."""
    agent_invocations = []
    llm_calls = []
    tool_executions = []
    for span in spans:
        if span.is_agent_invocation():
            agent_invocations.append(span)
        elif span.is_llm_call():
            llm_calls.append(span)
        elif span.is_tool_execution():
            tool_executions.append(span)
        else:
            msg = f"Unexpected span: {span}"
            raise AssertionError(msg)
    return agent_invocations, llm_calls, tool_executions
