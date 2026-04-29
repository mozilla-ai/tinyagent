from __future__ import annotations

import asyncio
import builtins
import inspect
import json
from typing import TYPE_CHECKING, Any, Self, overload

from any_llm import AnyLLM, LLMProvider
from any_llm.utils.aio import run_async_in_sync
from mcp.types import CallToolResult, TextContent
from opentelemetry import trace as otel_trace

from tinyagent.callbacks.context import Context
from tinyagent.callbacks.wrapper import _TinyAgentWrapper
from tinyagent.logging import logger
from tinyagent.tools.wrappers import _wrap_tools
from tinyagent.tracing.agent_trace import AgentTrace
from tinyagent.tracing.attributes import GenAI, TinyAgentAttributes
from tinyagent.utils.cast import safe_cast_argument

if TYPE_CHECKING:
    import types
    from collections.abc import Callable, Sequence

    from any_llm.types.completion import ChatCompletion
    from opentelemetry.trace import Tracer
    from pydantic import BaseModel

    from tinyagent.config import AgentConfig, Tool
    from tinyagent.serving import A2AServingConfig, MCPServingConfig, ServerHandle
    from tinyagent.tools.mcp.mcp_client import MCPClient


INSIDE_NOTEBOOK = hasattr(builtins, "__IPYTHON__")


DEFAULT_SYSTEM_PROMPT = """
You are an agent that uses tools (whenever they are available) to answer the user's query.

You will keep calling tools until you find a final answer.
Once you have a final answer, you MUST call the `final_answer` tool.

You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls.
""".strip()


class AgentCancel(Exception):  # noqa: N818
    """Abstract base class for control-flow exceptions raised in callbacks.

    Within a callback, raise an exception inherited from `AgentCancel` when you
    want to intentionally stop agent execution and handle that specific case in
    your application code.

    Unlike regular exceptions (which are wrapped in `AgentRunError`), `AgentCancel`
    subclasses propagate directly to the caller, allowing you to catch them by
    their specific type.

    When to use AgentCancel vs regular exceptions:
    - Use `AgentCancel`: When stopping execution is expected behavior
      (rate limits, safety guardrails, validation failures) and you
      want to handle it distinctly in your application.
    - Use regular exceptions: When something unexpected goes wrong,
      and you want consistent error handling via `AgentRunError`.

    Example:
        class StopOnLimit(AgentCancel):
            pass

        class LimitCallsCallback(Callback):
            def before_tool_execution(self, context, *args, **kwargs):
                if context.shared.get("call_count", 0) > 10:
                    raise StopOnLimit("Exceeded call limit")
                return context

        try:
            agent.run("prompt")
        except StopOnLimit as e:
            print(f"Canceled: {e}")
            print(f"Collected {len(e.trace.spans)} spans")
        except AgentRunError as e:
            print(f"Unexpected error: {e.original_exception}")

    """

    _trace: AgentTrace | None

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        if cls is AgentCancel:
            msg = "AgentCancel cannot be instantiated directly; subclass it instead"
            raise TypeError(msg)
        return super().__new__(cls)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._trace = None

    @property
    def trace(self) -> AgentTrace | None:
        """Execution trace collected before cancellation.

        Returns None if accessed before the framework processes the exception.
        """
        return self._trace


class AgentRunError(Exception):
    """Wrapper for unexpected exceptions that occur during agent execution.

    When an unexpected exception is raised during agent execution (from
    callbacks, tools, or the underlying loop), it is caught and wrapped in
    AgentRunError.

    Note: Exceptions that inherit from AgentCancel are not wrapped,
        they propagate directly to the caller.

    AgentRunError ensures:

    - The execution trace is preserved - you can inspect what happened
       before the error via the `trace` property.
    - Original exception access - the wrapped exception is available
       via `original_exception` for debugging.

    Catch this when you want access to the collected trace even on failure.

    Example:
        try:
            agent.run("prompt")
        except AgentRunError as e:
            print(f"Error: {e.original_exception}")
            print(f"Trace had {len(e.trace.spans)} spans before failure")

    """

    _trace: AgentTrace
    _original_exception: Exception

    def __init__(self, trace: AgentTrace, original_exception: Exception):
        self._trace = trace
        self._original_exception = original_exception
        super().__init__(str(original_exception))

    @property
    def trace(self) -> AgentTrace:
        """The execution trace collected up to the point of failure."""
        return self._trace

    @property
    def original_exception(self) -> Exception:
        """The underlying exception that was caught."""
        return self._original_exception

    def __str__(self) -> str:
        """Return the string representation of the original exception."""
        return str(self._original_exception)

    def __repr__(self) -> str:
        """Return the detailed representation of the AgentRunError."""
        return f"AgentRunError({self._original_exception!r})"


def _unwrap_agent_cancel(exc: BaseException) -> AgentCancel | None:
    """Traverse an exception chain to find an AgentCancel if present.

    Python's exception chaining stores the original exception in __cause__
    (explicit: `raise X from Y`) or __context__ (implicit: `raise X` inside
    an except block). This function walks that chain to find any AgentCancel.

    Args:
        exc: The exception to inspect.

    Returns:
        The first AgentCancel found in the exception chain, or None if the
        chain contains no AgentCancel instances.

    """
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, AgentCancel):
            return current
        current = current.__cause__ or current.__context__
    return None


class ToolExecutor:
    """Executor for tools that wraps tool functions to work with the MCP client."""

    def __init__(self, tool_function: Callable[..., Any]) -> None:
        """Initialize the tool executor.

        Args:
            tool_function: The tool function to execute

        """
        self.tool_function = tool_function

    async def call_tool(self, request: dict[str, Any]) -> str:
        """Call the tool function.

        Args:
            request: The tool request with name and arguments

        Returns:
            Tool execution result

        """
        try:
            arguments = request.get("arguments", {})

            if asyncio.iscoroutinefunction(self.tool_function):
                result = await self.tool_function(**arguments)
            else:
                result = self.tool_function(**arguments)

            if (
                isinstance(result, CallToolResult)
                and result.content
                and isinstance(result.content[0], TextContent)
            ):
                result = result.content[0].text
            return str(result)

        except Exception as e:
            return f"Error calling tool: {e}"


def final_answer(answer: str) -> str:
    """Return the final answer to the user."""
    return answer


class TinyAgent:
    """A lightweight agent implementation using `any-llm`.

    Modeled after the JS implementation https://huggingface.co/blog/tiny-agents.

    Use `TinyAgent.create(config)` (sync) or `TinyAgent.create_async(config)` to
    construct an agent that has had its tools loaded. The returned instance is
    callable via `.run(prompt)` / `.run_async(prompt)` and produces an
    `AgentTrace` with the final output and observability data.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the TinyAgent.

        Args:
            config: Agent configuration.

        """
        self.config = config

        self._mcp_clients: list[MCPClient] = []
        self._tools: list[Any] = []

        self._add_span_callbacks()
        self._wrapper = _TinyAgentWrapper()

        self._tracer: Tracer = otel_trace.get_tracer("tinyagent")

        self._lock = asyncio.Lock()
        self._callback_contexts: dict[int, Context] = {}

        self.clients: dict[str, ToolExecutor] = {}

        provider_name, model_id = AnyLLM.split_model_provider(self.config.model_id)
        if provider_name == "gateway":
            provider_name, model_id = AnyLLM.split_model_provider(model_id)
        self.uses_openai = provider_name == LLMProvider.OPENAI

        llm_kwargs: dict[str, Any] = dict(self.config.any_llm_args or {})
        if self.config.api_key:
            llm_kwargs["api_key"] = self.config.api_key
        if self.config.api_base:
            llm_kwargs["api_base"] = self.config.api_base
        self.llm = AnyLLM.create(provider_name, **llm_kwargs)

        self.completion_params: dict[str, Any] = {
            "model": model_id,
            "tools": [],
            "tool_choice": "required",
            **(self.config.model_args or {}),
        }

        if not self.uses_openai and self.completion_params["tool_choice"] == "required":
            self.config.tools.append(final_answer)

    @classmethod
    def create(cls, agent_config: AgentConfig) -> TinyAgent:
        """Create an agent with its tools loaded synchronously."""
        return run_async_in_sync(
            cls.create_async(agent_config=agent_config),
            allow_running_loop=INSIDE_NOTEBOOK,
        )

    @classmethod
    async def create_async(cls, agent_config: AgentConfig) -> TinyAgent:
        """Create an agent with its tools loaded asynchronously."""
        agent = cls(agent_config)
        await agent._load_agent()
        return agent

    async def _load_tools(self, tools: Sequence[Tool]) -> list[Any]:
        wrapped, mcp_clients = await _wrap_tools(tools)
        self._mcp_clients.extend(mcp_clients)
        return wrapped

    async def _load_agent(self) -> None:
        """Load the agent and its tools."""
        wrapped_tools = await self._load_tools(self.config.tools)

        self._tools = wrapped_tools

        for tool in wrapped_tools:
            tool_name = tool.__name__
            tool_desc = tool.__doc__ or f"Tool to {tool_name}"

            if not hasattr(tool, "__input_schema__"):
                sig = inspect.signature(tool)
                properties = {}
                required = []

                for param_name, param in sig.parameters.items():
                    if param.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ):
                        continue

                    properties[param_name] = {
                        "type": "string",
                        "description": f"Parameter {param_name}",
                    }

                    if param.default == inspect.Parameter.empty or self.uses_openai:
                        required.append(param_name)

                input_schema = {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            else:
                input_schema = tool.__input_schema__

            function_def = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_desc,
                    "parameters": input_schema,
                },
            }
            if self.uses_openai:
                function_def["function"]["parameters"]["additionalProperties"] = False  # type: ignore[index]
                function_def["function"]["strict"] = True  # type: ignore[index]

            self.completion_params["tools"].append(function_def)
            self.clients[tool_name] = ToolExecutor(tool)

    async def cleanup_async(self) -> None:
        """Clean up resources including MCP client connections.

        This should be called when you're done using the agent to ensure
        all resources are properly released.
        """
        for client in self._mcp_clients:
            await client.disconnect()
        self._mcp_clients.clear()

    async def __aenter__(self) -> Self:
        """Enter the async context manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Exit the async context manager and clean up resources."""
        await self.cleanup_async()

    def run(self, prompt: str | list[dict[str, Any]], **kwargs: Any) -> AgentTrace:
        """Run the agent with the given prompt."""
        return run_async_in_sync(
            self.run_async(prompt, **kwargs), allow_running_loop=INSIDE_NOTEBOOK
        )

    async def run_async(
        self, prompt: str | list[dict[str, Any]], **kwargs: Any
    ) -> AgentTrace:
        """Run the agent asynchronously with the given prompt.

        Args:
            prompt: The user prompt to be passed to the agent. Can be a plain
                string or a list of message dicts (e.g.
                ``[{"role": "user", "content": "hello"}]``) following the
                OpenAI chat-completion message format. When a list is provided
                it is forwarded directly to the underlying LLM, giving callers
                full control over the conversation structure.

            kwargs: Will be passed to the underlying runner.

        Returns:
            The `AgentTrace` containing information about the steps taken by
            the agent.

        """
        trace = AgentTrace()
        trace_id: int

        try:
            with self._tracer.start_as_current_span(
                f"invoke_agent [{self.config.name}]"
            ) as invoke_span:
                async with self._lock:
                    trace_id = invoke_span.get_span_context().trace_id
                    self._wrapper.callback_context[trace_id] = Context(
                        current_span=invoke_span,
                        trace=AgentTrace(),
                        tracer=self._tracer,
                        shared={},
                    )

                    if len(self._wrapper.callback_context) == 1:
                        await self._wrapper.wrap(agent=self)

                from tinyagent import __version__ as _TINYAGENT_VERSION  # noqa: N812

                invoke_span.set_attributes(
                    {
                        GenAI.OPERATION_NAME: "invoke_agent",
                        GenAI.AGENT_NAME: self.config.name,
                        GenAI.AGENT_DESCRIPTION: self.config.description
                        or "No description.",
                        GenAI.REQUEST_MODEL: self.config.model_id,
                        TinyAgentAttributes.VERSION: _TINYAGENT_VERSION,
                    }
                )

                context = self._wrapper.callback_context[trace_id]
                for callback in self.config.callbacks:
                    result = callback.before_agent_invocation(context, prompt, **kwargs)
                    if asyncio.iscoroutinefunction(callback.before_agent_invocation):
                        context = await result  # type: ignore[misc]
                    else:
                        context = result

                final_output = await self._run_async(prompt, **kwargs)

        except Exception as e:
            async with self._lock:
                if len(self._wrapper.callback_context) == 1:
                    await self._wrapper.unwrap(self)
                wrapped_context = self._wrapper.callback_context.pop(trace_id, None)
                if wrapped_context is not None:
                    trace = wrapped_context.trace
                    for callback in self.config.callbacks:
                        assert wrapped_context is not None
                        result = callback.after_agent_invocation(
                            wrapped_context, prompt, **kwargs
                        )
                        if asyncio.iscoroutinefunction(callback.after_agent_invocation):
                            wrapped_context = await result  # type: ignore[misc]
                        else:
                            wrapped_context = result

            trace.add_span(invoke_span)

            if isinstance(e, AgentCancel):
                e._trace = trace
                raise

            if cancel := _unwrap_agent_cancel(e):
                cancel._trace = trace
                raise cancel from e

            raise AgentRunError(trace, e) from e

        async with self._lock:
            if len(self._wrapper.callback_context) == 1:
                await self._wrapper.unwrap(self)
            wrapped_context = self._wrapper.callback_context.pop(trace_id, None)
            if wrapped_context is not None:
                trace = wrapped_context.trace
                for callback in self.config.callbacks:
                    assert wrapped_context is not None
                    result = callback.after_agent_invocation(
                        wrapped_context, prompt, **kwargs
                    )
                    if asyncio.iscoroutinefunction(callback.after_agent_invocation):
                        wrapped_context = await result  # type: ignore[misc]
                    else:
                        wrapped_context = result

        trace.add_span(invoke_span)
        trace.final_output = final_output
        return trace

    async def _run_async(
        self, prompt: str | list[dict[str, Any]], **kwargs: Any
    ) -> str | BaseModel:
        if self.uses_openai:
            self.completion_params["tool_choice"] = "auto"
            if self.config.output_type:
                self.completion_params["response_format"] = self.config.output_type

        if isinstance(prompt, list):
            messages = prompt
        else:
            messages = [
                {
                    "role": "system",
                    "content": self.config.instructions or DEFAULT_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]

        if kwargs.pop("max_turns", None):
            logger.warning(
                "`max_turns` is deprecated and has no effect. See https://docs.mozilla.ai/tinyagent/callbacks/#example-limit-the-number-of-steps"
            )

        while True:
            completion_params = self.completion_params.copy()

            completion_params["messages"] = messages

            response: ChatCompletion = await self.call_model(**completion_params)

            message = response.choices[0].message

            messages.append(message.model_dump())

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    f = tool_call.function  # type: ignore[union-attr]
                    tool_name = f.name
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "",
                        "name": tool_name,
                    }

                    if tool_name not in self.clients:
                        tool_message["content"] = (
                            f"Error calling tool: No tool found with name: {tool_name}"
                        )
                        messages.append(tool_message)
                        continue

                    tool_args = {}
                    if f.arguments:
                        tool_args = json.loads(f.arguments)

                    client = self.clients[tool_name]

                    if hasattr(client.tool_function, "__annotations__"):
                        func_args = client.tool_function.__annotations__
                        for arg_name, arg_type in func_args.items():
                            if arg_name in tool_args:
                                try:
                                    tool_args[arg_name] = safe_cast_argument(
                                        tool_args[arg_name], arg_type
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Failed to cast argument '{arg_name}': {e}"
                                    )

                    result = await client.call_tool(
                        {"name": tool_name, "arguments": tool_args}
                    )
                    tool_message["content"] = result
                    messages.append(tool_message)

                    if tool_name == "final_answer":
                        if self.config.output_type:
                            return await self._return_output_type(
                                str(result), completion_params
                            )
                        return str(result)

            elif message.role == "assistant" and message.content:
                if self.config.output_type:
                    return await self._return_output_type(
                        str(message.content), completion_params
                    )
                return str(message.content)

    async def _return_output_type(
        self, output: str, completion_params: dict[str, Any]
    ) -> str | BaseModel:
        if not self.config.output_type:
            return output

        if self.uses_openai:
            return self.config.output_type.model_validate_json(output)

        completion_params["messages"] = [
            {
                "role": "system",
                "content": "You are an expert that can convert raw text into structured JSON.",
            },
            {
                "role": "user",
                "content": f"Please conform this output:\n{output}\nTo match the following schema:\n{self.config.output_type.model_json_schema()}.",
            },
        ]

        completion_params["response_format"] = self.config.output_type
        if "tools" in completion_params:
            completion_params.pop("tools")
            completion_params.pop("tool_choice", None)
            completion_params.pop("parallel_tool_calls", None)
        response = await self.call_model(**completion_params)
        return self.config.output_type.model_validate_json(
            response.choices[0].message.content  # type: ignore[arg-type]
        )

    async def call_model(self, **completion_params: Any) -> ChatCompletion:
        return await self.llm.acompletion(**completion_params)  # type: ignore[no-any-return]

    async def update_output_type_async(
        self, output_type: type[BaseModel] | None
    ) -> None:
        """Update the output type of the agent in-place.

        Args:
            output_type: The new output type to use, or None to remove output type constraint

        """
        self.config.output_type = output_type

    def _add_span_callbacks(self) -> None:
        if self.config.callbacks is None:
            return

        from tinyagent.callbacks.span_end import SpanEndCallback
        from tinyagent.callbacks.span_generation import _SpanGeneration

        if not any(isinstance(c, _SpanGeneration) for c in self.config.callbacks):
            self.config.callbacks.insert(0, _SpanGeneration())
        if not any(isinstance(c, SpanEndCallback) for c in self.config.callbacks):
            self.config.callbacks.append(SpanEndCallback())

    async def _serve_a2a_async(
        self, serving_config: A2AServingConfig | None
    ) -> ServerHandle:
        from tinyagent.serving import (
            A2AServingConfig,
            _get_a2a_app_async,
            serve_a2a_async,
        )

        if serving_config is None:
            serving_config = A2AServingConfig()

        app = await _get_a2a_app_async(self, serving_config=serving_config)

        return await serve_a2a_async(
            app,
            host=serving_config.host,
            port=serving_config.port,
            endpoint=serving_config.endpoint,
            log_level=serving_config.log_level,
        )

    async def _serve_mcp_async(self, serving_config: MCPServingConfig) -> ServerHandle:
        from tinyagent.serving import serve_mcp_async

        return await serve_mcp_async(
            self,
            host=serving_config.host,
            port=serving_config.port,
            endpoint=serving_config.endpoint,
            log_level=serving_config.log_level,
        )

    @overload
    async def serve_async(self, serving_config: MCPServingConfig) -> ServerHandle: ...

    @overload
    async def serve_async(
        self, serving_config: A2AServingConfig | None = None
    ) -> ServerHandle: ...

    async def serve_async(
        self, serving_config: MCPServingConfig | A2AServingConfig | None = None
    ) -> ServerHandle:
        """Serve this agent asynchronously using the protocol defined in the serving_config.

        Args:
            serving_config: Configuration for serving the agent. If None, uses default A2AServingConfig.
                          Must be an instance of A2AServingConfig or MCPServingConfig.

        Returns:
            A ServerHandle instance that provides methods for managing the server lifecycle.

        Raises:
            ImportError: If the `a2a` dependencies are not installed and an `A2AServingConfig` is used.

        """
        from tinyagent.serving import MCPServingConfig

        if isinstance(serving_config, MCPServingConfig):
            return await self._serve_mcp_async(serving_config)
        return await self._serve_a2a_async(serving_config)
