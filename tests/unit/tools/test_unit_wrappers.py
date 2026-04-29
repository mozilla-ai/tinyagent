import asyncio

import pytest

from tinyagent.tools.wrappers import _wrap_tools


def test_bad_functions() -> None:
    """Test that `verify_callable` rejects functions missing required metadata."""

    def missing_return_type(foo: str):  # type: ignore[no-untyped-def]
        """Docstring for foo."""
        return foo

    with pytest.raises(ValueError, match="return type"):
        asyncio.run(_wrap_tools([missing_return_type]))

    def missing_docstring(foo: str) -> str:
        return foo

    with pytest.raises(ValueError, match="docstring"):
        asyncio.run(_wrap_tools([missing_docstring]))

    def missing_param_type(foo) -> str:  # type: ignore[no-untyped-def]
        """Docstring for foo."""
        return foo  # type: ignore[no-any-return]

    with pytest.raises(ValueError, match="typed arguments"):
        asyncio.run(_wrap_tools([missing_param_type]))

    def good_function(foo: str) -> str:
        """Docstring for foo.

        Args:
            foo: The foo argument.

        Returns:
            The foo result.
        """
        return foo

    asyncio.run(_wrap_tools([good_function]))


def test_wrap_tool_passes_through_callables() -> None:
    """A plain callable should pass through `_wrap_tools` and remain callable."""

    def my_tool(x: str) -> str:
        """Echo x."""
        return x

    wrapped, mcp_clients = asyncio.run(_wrap_tools([my_tool]))
    assert mcp_clients == []
    assert len(wrapped) == 1
    assert wrapped[0]("hello") == "hello"


def test_wrap_tool_invalid_type_raises() -> None:
    with pytest.raises(ValueError, match="MCPStdio"):
        asyncio.run(_wrap_tools([42]))
