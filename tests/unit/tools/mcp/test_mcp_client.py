import inspect
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock

import pytest

from tinyagent.config import MCPStdio
from tinyagent.tools.mcp.mcp_client import MCPClient


@pytest.mark.asyncio
async def test_create_tool_function_with_complex_schema() -> None:
    """Test the core tool function creation with comprehensive parameter handling."""
    from mcp.types import Tool as MCPTool

    mock_session = AsyncMock()
    mock_result = AsyncMock()
    mock_result.content = [AsyncMock()]
    mock_result.content[0].text = "Tool executed successfully"
    mock_session.call_tool.return_value = mock_result

    config = MCPStdio(command="test", args=[])
    client = MCPClient(config=config)
    client._session = mock_session

    complex_tool = MCPTool(
        name="complex_search",
        description="Search with multiple parameter types",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string"},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results",
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Include result metadata",
                },
                "filters": {"type": "object", "description": "Search filters"},
                "tags": {"type": "array", "description": "Filter tags"},
                "threshold": {"type": "number", "description": "Similarity threshold"},
                "optional_param": {
                    "type": "string",
                    "description": "Optional parameter",
                },
            },
            "required": ["query", "max_results", "include_metadata"],
        },
    )

    tool_func = client._create_tool_function(complex_tool)

    assert tool_func.__name__ == "complex_search"
    assert tool_func.__doc__ is not None
    assert "Search with multiple parameter types" in tool_func.__doc__
    assert "query: Search query string" in tool_func.__doc__
    assert "max_results: Maximum number of results" in tool_func.__doc__

    sig = inspect.signature(tool_func)
    params = sig.parameters

    assert "query" in params
    assert params["query"].annotation is str
    assert params["query"].default is inspect.Parameter.empty

    assert "max_results" in params
    assert params["max_results"].annotation is int
    assert params["max_results"].default is inspect.Parameter.empty

    assert "include_metadata" in params
    assert params["include_metadata"].annotation is bool
    assert params["include_metadata"].default is inspect.Parameter.empty

    assert "optional_param" in params
    assert params["optional_param"].annotation == Optional[str]  # noqa: UP045
    assert params["optional_param"].default is None

    assert params["filters"].annotation == Optional[dict]  # noqa: UP045
    assert params["tags"].annotation == Optional[list]  # noqa: UP045
    assert params["threshold"].annotation == Optional[float]  # noqa: UP045

    assert sig.return_annotation is str

    result = await tool_func(query="test query", max_results=10, include_metadata=True)
    assert result == "Tool executed successfully"
    mock_session.call_tool.assert_called_with(
        "complex_search",
        {"query": "test query", "max_results": 10, "include_metadata": True},
    )

    await tool_func(
        query="another query",
        max_results=5,
        include_metadata=False,
        optional_param="optional_value",
        threshold=0.8,
    )
    mock_session.call_tool.assert_called_with(
        "complex_search",
        {
            "query": "another query",
            "max_results": 5,
            "include_metadata": False,
            "optional_param": "optional_value",
            "threshold": 0.8,
        },
    )

    client._session = None
    error_result = await tool_func(query="test", max_results=1, include_metadata=True)
    assert "Error: MCP session not available" in error_result


def test_mcp_client_import_failure_raises_import_error(monkeypatch) -> None:
    """Failed mcp import must raise ImportError from MCPClient, not NameError (#22)."""
    import importlib.util
    import sys
    import types

    # Build a fake mcp package where streamablehttp_client is missing (mcp>=2 shape).
    fake_mcp = types.ModuleType("mcp")
    fake_client = types.ModuleType("mcp.client")
    fake_sse = types.ModuleType("mcp.client.sse")
    fake_stdio = types.ModuleType("mcp.client.stdio")
    fake_http = types.ModuleType("mcp.client.streamable_http")
    fake_types = types.ModuleType("mcp.types")

    fake_sse.sse_client = object()
    fake_stdio.stdio_client = object()
    # Intentionally omit streamablehttp_client so the real import fails.

    class _ClientSession:
        pass

    class _StdioServerParameters:
        pass

    class _Tool:
        pass

    fake_mcp.ClientSession = _ClientSession
    fake_mcp.StdioServerParameters = _StdioServerParameters
    fake_types.Tool = _Tool

    modules = {
        "mcp": fake_mcp,
        "mcp.client": fake_client,
        "mcp.client.sse": fake_sse,
        "mcp.client.stdio": fake_stdio,
        "mcp.client.streamable_http": fake_http,
        "mcp.types": fake_types,
    }
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)

    src = Path(__file__).resolve().parents[4] / "src/tinyagent/tools/mcp/mcp_client.py"
    # When running from a checkout the path above works; fall back to package file.
    if not src.is_file():
        import tinyagent.tools.mcp.mcp_client as installed

        src = Path(installed.__file__)

    spec = importlib.util.spec_from_file_location(
        "tinyagent_mcp_client_import_guard_test",
        src,
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod.missing_mcp_error is not None

    config = MCPStdio(command="test", args=[])
    with pytest.raises(ImportError, match="MCP support requires") as exc_info:
        mod.MCPClient(config=config)
    assert exc_info.value.__cause__ is mod.missing_mcp_error
