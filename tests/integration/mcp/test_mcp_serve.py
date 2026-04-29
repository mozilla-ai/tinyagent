import datetime
import os

import pytest
from sse_starlette.sse import AppStatus

from tinyagent import AgentConfig, TinyAgent
from tinyagent.config import MCPSse
from tinyagent.serving import MCPServingConfig
from tinyagent.testing.helpers import (
    DEFAULT_SMALL_MODEL_ID,
    get_default_agent_model_args,
    wait_for_server_async,
)
from tinyagent.tracing.agent_trace import AgentTrace
from tinyagent.tracing.attributes import GenAI


pytestmark = pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"), reason="MISTRAL_API_KEY not set"
)


def _assert_valid_agent_trace(agent_trace: AgentTrace) -> None:
    assert isinstance(agent_trace, AgentTrace)
    assert agent_trace.final_output


def _assert_contains_current_date_info(final_output: str) -> None:
    now = datetime.datetime.now()
    assert all(
        [
            str(now.year) in final_output,
            str(now.day) in final_output,
            now.strftime("%B") in final_output,
        ]
    )


def _assert_has_tool_date_agent_call(agent_trace: AgentTrace) -> None:
    assert any(
        span.is_tool_execution()
        and span.attributes.get(GenAI.TOOL_NAME, None) == "as-tool-date_agent"
        for span in agent_trace.spans
    )


DATE_PROMPT = (
    "What date and time is it right now? "
    "In your answer please include the year, month, day, and time. "
    "Example answer could be something like 'Today is December 15, 2024'"
)


@pytest.mark.asyncio
async def test_mcp_serve(test_port: int) -> None:
    """Test serving a TinyAgent over MCP and using it as a tool from another TinyAgent."""
    main_agent = None
    server_handle = None

    try:
        tool_agent_endpoint = "tool_agent"

        def get_datetime() -> str:
            """Return the current date and time."""
            return str(datetime.datetime.now())

        date_agent_cfg = AgentConfig(
            instructions="Use the available tools to obtain additional information to answer the query.",
            name="date_agent",
            model_id=DEFAULT_SMALL_MODEL_ID,
            description="Agent that can return the current date.",
            tools=[get_datetime],
            model_args=get_default_agent_model_args(),
        )
        date_agent = await TinyAgent.create_async(date_agent_cfg)

        server_url = f"http://localhost:{test_port}/{tool_agent_endpoint}/sse"
        server_handle = await date_agent.serve_async(
            serving_config=MCPServingConfig(
                port=test_port,
                endpoint=f"/{tool_agent_endpoint}",
                log_level="info",
            )
        )
        ping_url = f"http://localhost:{test_port}"
        await wait_for_server_async(ping_url)

        main_agent_cfg = AgentConfig(
            model_id=DEFAULT_SMALL_MODEL_ID,
            instructions="Use the available tools to obtain additional information to answer the query.",
            description="The orchestrator that uses the date agent via MCP.",
            tools=[MCPSse(url=server_url, client_session_timeout_seconds=300)],
            model_args=get_default_agent_model_args(),
        )

        main_agent = await TinyAgent.create_async(main_agent_cfg)

        agent_trace = await main_agent.run_async(DATE_PROMPT)

        _assert_valid_agent_trace(agent_trace)
        _assert_contains_current_date_info(str(agent_trace.final_output))
        _assert_has_tool_date_agent_call(agent_trace)

    finally:
        if main_agent:
            for mcp_client in main_agent._mcp_clients:
                await mcp_client.disconnect()
        AppStatus.should_exit = True
        if server_handle:
            await server_handle.shutdown()
