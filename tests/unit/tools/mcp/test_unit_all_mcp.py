from collections.abc import Sequence

import pytest

from tinyagent.config import MCPSse
from tinyagent.tools import MCPClient, _wrap_tools


@pytest.mark.asyncio
async def test_mcp_tool_wrapping(tools: Sequence[str]) -> None:
    """MCP tools should be loaded via the connect/list_tools path and wrapped as callables."""
    from typing import Any
    from unittest.mock import AsyncMock, patch

    def create_mock_tool(tool_name: str) -> Any:
        def mock_tool() -> str:
            """Mock tool for testing."""
            return f"mock_result_{tool_name}"

        mock_tool.__name__ = f"mock_tool_{tool_name}"
        mock_tool.__doc__ = f"Mock tool for {tool_name}."
        return mock_tool

    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.list_tools = AsyncMock(
        return_value=[create_mock_tool(str(tool)) for tool in tools]
    )

    with patch("tinyagent.tools.wrappers.MCPClient", return_value=mock_client):
        mcp_config = MCPSse(
            url="http://localhost:8000/sse", tools=[str(tool) for tool in tools]
        )
        wrapped_tools, mcp_clients = await _wrap_tools([mcp_config])

        assert len(mcp_clients) == 1
        assert len(wrapped_tools) >= len(tools)
        mock_client.connect.assert_called_once()


@pytest.mark.asyncio
async def test_mcp_client_tool_filtering(
    mcp_client: MCPClient,
    tools: Sequence[str],
) -> None:
    """Test the _filter_tools method of MCPClient."""
    from mcp.types import Tool as MCPTool

    all_tools = [
        MCPTool(name=tool, inputSchema={"type": "object", "properties": {}})
        for tool in ["tool1", "tool2", "tool3"]
    ]

    mcp_client.config = mcp_client.config.model_copy(update={"tools": None})
    filtered_tools = mcp_client._filter_tools(all_tools)
    assert len(filtered_tools) == 3

    mcp_client.config = mcp_client.config.model_copy(
        update={"tools": ["tool3", "tool1"]}
    )
    filtered_tools = mcp_client._filter_tools(all_tools)
    assert len(filtered_tools) == 2
    assert filtered_tools[0].name == "tool3"
    assert filtered_tools[1].name == "tool1"

    mcp_client.config = mcp_client.config.model_copy(
        update={"tools": ["tool1", "nonexistent"]}
    )
    with pytest.raises(ValueError) as excinfo:  # noqa: PT011
        mcp_client._filter_tools(all_tools)
    assert "Missing: ['nonexistent']" in str(excinfo.value)


def test_mcp_client_tool_conversion(mcp_client: MCPClient) -> None:
    """Test converting MCP tools to callable functions."""
    from mcp.types import Tool as MCPTool

    test_tool = MCPTool(
        name="test_tool",
        description="A test tool",
        inputSchema={
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "Test parameter"},
                "param2": {"type": "integer", "description": "Another parameter"},
            },
            "required": ["param1"],
        },
    )

    callable_tools = mcp_client._convert_tools_to_callables([test_tool])
    assert len(callable_tools) == 1

    tool_func = callable_tools[0]
    assert callable(tool_func)

    result = tool_func(param1="test")
    assert "test_tool" in result
