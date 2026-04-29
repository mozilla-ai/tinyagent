"""TinyAgent calling another TinyAgent over A2A as a tool."""

import os
from typing import Any

import pytest

pytest.importorskip("a2a")

from tinyagent import AgentConfig, TinyAgent
from tinyagent.callbacks import Callback, Context
from tinyagent.serving import A2AServingConfig
from tinyagent.testing.helpers import (
    DEFAULT_HTTP_KWARGS,
    DEFAULT_SMALL_MODEL_ID,
    get_default_agent_model_args,
)
from tinyagent.tools import a2a_tool_async
from tinyagent.tracing.agent_trace import AgentTrace
from tinyagent.tracing.attributes import GenAI

from .conftest import (
    DATE_PROMPT,
    a2a_client_from_agent,
    assert_contains_current_date_info,
    get_datetime,
)

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


def _assert_valid_agent_trace(agent_trace: AgentTrace) -> None:
    assert isinstance(agent_trace, AgentTrace)
    assert agent_trace.final_output


def _assert_has_date_agent_tool_call(agent_trace: AgentTrace) -> None:
    assert any(
        span.is_tool_execution()
        and span.attributes.get(GenAI.TOOL_NAME, None) == "call_date_agent"
        for span in agent_trace.spans
    )


@pytest.mark.asyncio
async def test_a2a_tool_async() -> None:
    """A TinyAgent contacts another TinyAgent over A2A using the adapter tool."""
    date_agent_cfg = AgentConfig(
        instructions="Use the available tools to obtain additional information to answer the query.",
        name="date_agent",
        model_id=DEFAULT_SMALL_MODEL_ID,
        description="Agent that can return the current date.",
        tools=[get_datetime],
        model_args=get_default_agent_model_args(),
        callbacks=[LimitLLMCalls(max_llm_calls=25)],
    )
    date_agent = await TinyAgent.create_async(date_agent_cfg)

    serving_config = A2AServingConfig(port=0, endpoint="/tool_agent", log_level="info")

    async with a2a_client_from_agent(date_agent, serving_config) as (_, server_url):
        main_agent_cfg = AgentConfig(
            instructions="Use the available tools to obtain additional information to answer the query.",
            description="The orchestrator that uses other agents via A2A.",
            model_id=DEFAULT_SMALL_MODEL_ID,
            tools=[await a2a_tool_async(server_url, http_kwargs=DEFAULT_HTTP_KWARGS)],
            model_args=get_default_agent_model_args(),
            callbacks=[LimitLLMCalls(max_llm_calls=25)],
        )

        main_agent = await TinyAgent.create_async(main_agent_cfg)

        agent_trace = await main_agent.run_async(DATE_PROMPT)

        _assert_valid_agent_trace(agent_trace)
        assert_contains_current_date_info(str(agent_trace.final_output))
        _assert_has_date_agent_tool_call(agent_trace)
