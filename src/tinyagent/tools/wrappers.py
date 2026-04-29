import asyncio
import inspect
from collections.abc import Callable, Sequence
from functools import wraps
from typing import Any, TypeVar

from tinyagent.config import MCPParams
from tinyagent.tools.mcp import MCPClient


def _wrap_no_exception(tool: Any) -> Any:
    @wraps(tool)
    def wrapped_function(*args: Any, **kwargs: Any) -> Any:
        try:
            return tool(*args, **kwargs)
        except Exception as e:
            return f"Error calling tool: {e}"

    @wraps(tool)
    async def wrapped_coroutine(*args: Any, **kwargs: Any) -> Any:
        try:
            return await tool(*args, **kwargs)
        except Exception as e:
            return f"Error calling tool: {e}"

    if asyncio.iscoroutinefunction(tool):
        return wrapped_coroutine

    return wrapped_function


def verify_callable(tool: Callable[..., Any]) -> None:
    """Verify that `tool` is a valid callable.

    - It needs to have some sort of docstring that describes what it does
    - It needs to have typed argument
    - It needs to have a typed return.

    We need these things because this info gets provided to the agent so that they know how and when to call the tool.
    """
    signature = inspect.signature(tool)
    if not tool.__doc__:
        msg = f"Tool {tool} needs to have a docstring but does not"
        raise ValueError(msg)

    if signature.return_annotation is inspect.Signature.empty:
        msg = f"Tool {tool} needs to have a return type but does not"
        raise ValueError(msg)
    for param in signature.parameters.values():
        if param.annotation is inspect.Signature.empty:
            msg = f"Tool {tool} needs to have typed arguments but does not"
            raise ValueError(msg)


T_co = TypeVar("T_co", covariant=True)


async def _wrap_tools(
    tools: Sequence[T_co],
) -> tuple[list[T_co], list[MCPClient]]:
    wrapped_tools = list[T_co]()
    mcp_clients: list[MCPClient] = []

    for tool in tools:
        if isinstance(tool, MCPParams):
            mcp_client = MCPClient(config=tool)

            await mcp_client.connect()

            callable_tools = await mcp_client.list_tools()

            for callable_tool in callable_tools:
                wrapped_tools.append(_wrap_no_exception(callable_tool))

            mcp_clients.append(mcp_client)
        elif callable(tool):
            verify_callable(tool)
            wrapped_tools.append(_wrap_no_exception(tool))
        else:
            msg = f"Tool {tool} needs to be of type `MCPStdio` or `callable` but is {type(tool)}"
            raise ValueError(msg)

    return wrapped_tools, mcp_clients
