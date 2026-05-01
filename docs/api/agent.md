---
title: Agent
description: TinyAgent, AgentCancel, and AgentRunError reference
---

## TinyAgent

A lightweight agent implementation using `any-llm`.

Modeled after the JS implementation https://huggingface.co/blog/tiny-agents.

Use `TinyAgent.create(config)` (sync) or `TinyAgent.create_async(config)` to construct an agent that has had its tools loaded. The returned instance is callable via `.run(prompt)` / `.run_async(prompt)` and produces an `AgentTrace` with the final output and observability data.

### `TinyAgent.create()`

Create an agent with its tools loaded synchronously.

```python
def create(
    agent_config: AgentConfig,
) -> TinyAgent
```

| Parameter | Type | Default |
|-----------|------|---------|
| `agent_config` | `AgentConfig` | *required* |

### `TinyAgent.create_async()`

Async variant of `create()` with the same parameters.

```python
async def create_async(
    agent_config: AgentConfig,
) -> TinyAgent
```

### `TinyAgent.run()`

Run the agent with the given prompt.

```python
def run(
    self,
    prompt: str | list[dict[str, Any]],
    **kwargs: Any,
) -> AgentTrace
```

| Parameter | Type | Default |
|-----------|------|---------|
| `prompt` | `str \| list[dict[str, Any]]` | *required* |
| `**kwargs` | `Any` | *required* |

### `TinyAgent.run_async()`

Async variant of `run()` with the same parameters.

```python
async def run_async(
    self,
    prompt: str | list[dict[str, Any]],
    **kwargs: Any,
) -> AgentTrace
```

### `TinyAgent.serve_async()`

Serve this agent asynchronously using the protocol defined in the serving_config.

```python
async def serve_async(
    self,
    serving_config: MCPServingConfig | A2AServingConfig | None = None,
) -> ServerHandle
```

### `TinyAgent.cleanup_async()`

Clean up resources (MCP connections, etc.). Called automatically when using the async context manager pattern.

---

## AgentCancel

Abstract base class for control-flow exceptions raised in callbacks.

Within a callback, raise an exception inherited from `AgentCancel` when you want to intentionally stop agent execution and handle that specific case in your application code.

Unlike regular exceptions (which are wrapped in `AgentRunError`), `AgentCancel` subclasses propagate directly to the caller, allowing you to catch them by their specific type.

When to use AgentCancel vs regular exceptions:
- Use `AgentCancel`: When stopping execution is expected behavior
(rate limits, safety guardrails, validation failures) and you
want to handle it distinctly in your application.
- Use regular exceptions: When something unexpected goes wrong,
and you want consistent error handling via `AgentRunError`.

**Example:**

```python
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
```

**Properties:**

- `trace` - `AgentTrace | None`: Execution trace collected before cancellation.

---

## AgentRunError

Wrapper for unexpected exceptions that occur during agent execution.

When an unexpected exception is raised during agent execution (from callbacks, tools, or the underlying loop), it is caught and wrapped in AgentRunError.

Note: Exceptions that inherit from AgentCancel are not wrapped, they propagate directly to the caller.

AgentRunError ensures:

- The execution trace is preserved - you can inspect what happened
before the error via the `trace` property.
- Original exception access - the wrapped exception is available
via `original_exception` for debugging.

Catch this when you want access to the collected trace even on failure.

**Example:**

```python
try:
    agent.run("prompt")
except AgentRunError as e:
    print(f"Error: {e.original_exception}")
    print(f"Trace had {len(e.trace.spans)} spans before failure")
```

**Properties:**

- `trace` - `AgentTrace`: The execution trace collected up to failure point.
- `original_exception` - `Exception`: The underlying exception that was caught.
