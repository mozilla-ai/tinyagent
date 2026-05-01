---
title: Tracing
description: AgentTrace, AgentSpan, CostInfo, TokenInfo, and GenAI reference
---

## AgentTrace

A trace that can be exported to JSON or printed to the console.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `spans` | `list[AgentSpan]` | A list of `AgentSpan` that form the trace. |
| `final_output` | `str \| dict[str, Any] \| BaseModel \| None` | Contains the final output message returned by the agent. |

### Properties

- `duration` - `timedelta`: Duration of the agent invocation span.
- `tokens` - `TokenInfo`: Total token usage across all LLM calls.
- `cost` - `CostInfo`: Total cost across all LLM calls.
- `final_output` - `str | None`: The final answer returned by the agent.

---

## AgentSpan

A span that can be exported to JSON or printed to the console.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` |  |
| `kind` | `SpanKind` |  |
| `parent` | `tracing.otel_types.SpanContext \| None` |  |
| `start_time` | `int \| None` |  |
| `end_time` | `int \| None` |  |
| `status` | `Status` |  |
| `context` | `SpanContext` |  |
| `attributes` | `dict[str, Any]` |  |
| `links` | `list[Link]` |  |
| `events` | `list[Event]` |  |
| `resource` | `Resource` |  |

### Methods

#### `AgentSpan.is_agent_invocation()`

Check whether this span is an agent invocation (the very first span).

```python
def is_agent_invocation(
    self,
) -> bool
```

#### `AgentSpan.is_llm_call()`

Check whether this span is a call to an LLM.

```python
def is_llm_call(
    self,
) -> bool
```

#### `AgentSpan.is_tool_execution()`

Check whether this span is an execution of a tool.

```python
def is_tool_execution(
    self,
) -> bool
```

---

## CostInfo

| Field | Type | Description |
|-------|------|-------------|
| `input_cost` | `float` |  |
| `output_cost` | `float` | Cost associated to the output tokens. |

**Properties:** `total_cost` - `float`: Total cost (input + output).

---

## TokenInfo

| Field | Type | Description |
|-------|------|-------------|
| `input_tokens` | `int` | Number of input tokens. |
| `output_tokens` | `int` | Number of output tokens. |

**Properties:** `total_tokens` - `int`: Total tokens (input + output).

---

## GenAI

Constants for accessing span attributes following OpenTelemetry semantic conventions for generative AI systems.

### Attributes

| Attribute | Value |
|-----------|-------|
| `GenAI.AGENT_DESCRIPTION` | `gen_ai.agent.description` |
| `GenAI.AGENT_NAME` | `gen_ai.agent.name` |
| `GenAI.INPUT_MESSAGES` | `gen_ai.input.messages` |
| `GenAI.OPERATION_NAME` | `gen_ai.operation.name` |
| `GenAI.OUTPUT` | `gen_ai.output` |
| `GenAI.OUTPUT_TYPE` | `gen_ai.output.type` |
| `GenAI.REQUEST_MODEL` | `gen_ai.request.model` |
| `GenAI.TOOL_ARGS` | `gen_ai.tool.args` |
| `GenAI.TOOL_DESCRIPTION` | `gen_ai.tool.description` |
| `GenAI.TOOL_NAME` | `gen_ai.tool.name` |
| `GenAI.USAGE_INPUT_COST` | `gen_ai.usage.input_cost` |
| `GenAI.USAGE_INPUT_TOKENS` | `gen_ai.usage.input_tokens` |
| `GenAI.USAGE_OUTPUT_COST` | `gen_ai.usage.output_cost` |
| `GenAI.USAGE_OUTPUT_TOKENS` | `gen_ai.usage.output_tokens` |
