# mypy: disable-error-code="method-assign,no-untyped-def"
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from opentelemetry.trace import Status, StatusCode

from tinyagent.callbacks.base import Callback
from tinyagent.tracing.attributes import GenAI

if TYPE_CHECKING:
    from any_llm.types.completion import (
        ChatCompletion,
        ChatCompletionMessageToolCall,
        CompletionUsage,
    )

    from tinyagent.callbacks.context import Context


class _SpanGeneration(Callback):
    """Generate OTel spans for each LLM call and tool execution in the TinyAgent loop.

    This callback is attached automatically by `TinyAgent` unless one is already
    present in the user-provided callback list.
    """

    def __init__(self) -> None:
        self.first_llm_calls: set[int] = set()

    def _serialize_for_attribute(self, data: Any) -> str:
        if isinstance(data, str):
            return data
        try:
            return json.dumps(data, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(data)

    def _determine_output_type(self, output: Any) -> str:
        if isinstance(output, str):
            try:
                json.loads(output)
            except json.JSONDecodeError:
                return "text"
        return "json"

    def _set_llm_input(
        self, context: Context, model_id: str, input_messages: list[dict[str, str]]
    ) -> Context:
        tracer = context.tracer
        span = tracer.start_span(f"call_llm {model_id}")

        span.set_attributes(
            {
                GenAI.OPERATION_NAME: "call_llm",
                GenAI.REQUEST_MODEL: model_id,
            }
        )

        trace_id = span.get_span_context().trace_id
        if trace_id not in self.first_llm_calls:
            self.first_llm_calls.add(trace_id)
            serialized_messages = self._serialize_for_attribute(input_messages)
            span.set_attribute(GenAI.INPUT_MESSAGES, serialized_messages)

        context.current_span = span
        return context

    def _set_llm_output(
        self,
        context: Context,
        output: str | list[dict[str, str]],
        input_tokens: int,
        output_tokens: int,
    ) -> Context:
        span = context.current_span
        output_type = self._determine_output_type(output)
        output_attr = self._serialize_for_attribute(output)

        span.set_attributes(
            {
                GenAI.OUTPUT: output_attr,
                GenAI.OUTPUT_TYPE: output_type,
                GenAI.USAGE_INPUT_TOKENS: input_tokens,
                GenAI.USAGE_OUTPUT_TOKENS: output_tokens,
            }
        )

        span.set_status(StatusCode.OK)
        return context

    def _set_tool_input(
        self,
        context: Context,
        name: str,
        description: str | None = None,
        args: dict[str, Any] | None = None,
        call_id: str | None = None,
    ) -> Context:
        tracer = context.tracer
        span = tracer.start_span(f"execute_tool {name}")

        attributes = {
            GenAI.OPERATION_NAME: "execute_tool",
            GenAI.TOOL_NAME: name,
        }

        if description is not None:
            attributes[GenAI.TOOL_DESCRIPTION] = description
        if args is not None:
            attributes[GenAI.TOOL_ARGS] = self._serialize_for_attribute(args)
        if call_id is not None:
            attributes["gen_ai.tool.call.id"] = call_id

        span.set_attributes(attributes)
        context.current_span = span
        return context

    def _determine_tool_status(
        self, tool_output: str, output_type: str
    ) -> Status | StatusCode:
        if output_type == "text" and "Error calling tool:" in tool_output:
            return Status(StatusCode.ERROR, description=tool_output)
        return StatusCode.OK

    def _set_tool_output(self, context: Context, tool_output: Any) -> Context:
        span = context.current_span

        if tool_output is None:
            tool_output = "{}"

        output_type = self._determine_output_type(tool_output)
        output_attr = self._serialize_for_attribute(tool_output)
        status = self._determine_tool_status(output_attr, output_type)

        span.set_attributes({GenAI.OUTPUT: output_attr, GenAI.OUTPUT_TYPE: output_type})
        span.set_status(status)
        return context

    def before_llm_call(self, context: Context, *args, **kwargs) -> Context:
        return self._set_llm_input(
            context,
            model_id=kwargs.get("model", "No model"),
            input_messages=kwargs.get("messages", []),
        )

    def after_llm_call(self, context: Context, *args, **kwargs) -> Context:
        response: ChatCompletion = args[0]

        if not response.choices:
            return context

        message = getattr(response.choices[0], "message", None)
        if not message:
            return context

        output: str | list[dict[str, str]] = ""
        if content := getattr(message, "content", None):
            output = content

        tool_calls: list[ChatCompletionMessageToolCall] | None
        if tool_calls := getattr(message, "tool_calls", None):
            output = [
                {
                    "tool.name": getattr(tool_call.function, "name", "No name"),
                    "tool.args": getattr(tool_call.function, "arguments", "No name"),
                }
                for tool_call in tool_calls
                if tool_call.function
            ]

        input_tokens = 0
        output_tokens = 0
        token_usage: CompletionUsage | None

        if token_usage := getattr(response, "usage", None):
            if token_usage:
                input_tokens = token_usage.prompt_tokens
                output_tokens = token_usage.completion_tokens

        return self._set_llm_output(context, output, input_tokens, output_tokens)

    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        request: dict[str, Any] = args[0]

        return self._set_tool_input(
            context,
            name=request.get("name", "No name"),
            args=request.get("arguments", {}),
        )

    def after_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        return self._set_tool_output(context, args[0])
