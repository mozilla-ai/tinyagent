<div align="center">

# tinyagent

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyPI](https://img.shields.io/pypi/v/mozilla-ai-tinyagent)](https://pypi.org/project/mozilla-ai-tinyagent/)

A minimal agent framework with first-class tracing, callbacks, MCP, and serving.

</div>

## What this is

`tinyagent` is a small, dependency-light Python implementation of the [HuggingFace Tiny Agents](https://huggingface.co/blog/tiny-agents) loop, built on [`any-llm`](https://github.com/mozilla-ai/any-llm). It gives you:

- A simple agent loop you can read in one sitting.
- Native [MCP](https://modelcontextprotocol.io/) tool support (stdio, SSE, streamable HTTP).
- OpenTelemetry-based tracing of LLM calls and tool executions, including token counts and cost.
- A callback system for guardrails, metrics, and intentional cancellation.
- Optional A2A and MCP serving so your agent can run as a service.

This package was extracted from [`any-agent`](https://github.com/mozilla-ai/any-agent), which still uses `tinyagent` under the hood as one of its supported framework backends.

## Install

```bash
pip install mozilla-ai-tinyagent
```

The PyPI distribution name is `mozilla-ai-tinyagent`; the import name is `tinyagent`.

Optional extras:

```bash
pip install 'mozilla-ai-tinyagent[a2a]'       # A2A serving
pip install 'mozilla-ai-tinyagent[composio]'  # Composio tools
pip install 'mozilla-ai-tinyagent[all]'       # everything
```

## Quickstart

```python
from tinyagent import TinyAgent, AgentConfig
from tinyagent.tools import search_web, visit_webpage

agent = TinyAgent.create(
    AgentConfig(
        model_id="mistral:mistral-small-latest",
        instructions="Use the tools to find an answer.",
        tools=[search_web, visit_webpage],
    )
)

trace = agent.run("Which agent framework is the simplest?")
print(trace.final_output)
```

`model_id` follows the [`any-llm` provider syntax](https://mozilla-ai.github.io/any-llm/providers/) (`provider:model`). Set the relevant API key (e.g. `MISTRAL_API_KEY`, `OPENAI_API_KEY`) in your environment.

## Use MCP tools

```python
from tinyagent import TinyAgent, AgentConfig
from tinyagent.config import MCPStdio

agent = TinyAgent.create(
    AgentConfig(
        model_id="mistral:mistral-small-latest",
        instructions="Use the available tools to answer.",
        tools=[
            MCPStdio(command="uvx", args=["duckduckgo-mcp-server"]),
        ],
    )
)

trace = agent.run("What is the capital of Pennsylvania?")
print(trace.final_output)
```

## Tracing

Every `agent.run(...)` returns an `AgentTrace` with the final output, the spans, token counts, and cost.

```python
trace = agent.run("...")
print(trace.duration)
print(trace.tokens)
print(trace.cost)
for span in trace.spans:
    print(span.name, span.attributes)
```

## Callbacks

Subclass `Callback` to observe or control execution. Each hook receives a `Context` and returns it, optionally mutating shared state.

```python
from tinyagent.callbacks import Callback, Context

class LimitToolCalls(Callback):
    def before_tool_execution(self, context: Context, *args, **kwargs) -> Context:
        context.shared["count"] = context.shared.get("count", 0) + 1
        if context.shared["count"] > 5:
            raise StopIteration("Too many tool calls")
        return context
```

See [`AgentCancel`](src/tinyagent/agent.py) for cancellation that preserves the trace.

## Evaluate

`tinyagent` ships two evaluators in `tinyagent.evaluation` for grading agent runs against criteria.

`LlmJudge` answers a question about a fixed context with a single LLM call:

```python
from tinyagent.evaluation import LlmJudge

judge = LlmJudge(model_id="mistral:mistral-small-latest")
result = judge.run(
    context=trace.final_output,
    question="Does the answer cite a primary source?",
)
print(result.passed, result.reasoning)
```

`AgentJudge` runs an agent that has tools to inspect the full `AgentTrace` (token counts, spans, tool calls), so it can answer richer questions:

```python
from tinyagent.evaluation import AgentJudge

judge = AgentJudge(model_id="mistral:mistral-small-latest")
eval_trace = judge.run(
    trace=trace,
    question="Did the agent use the search_web tool before answering?",
)
print(eval_trace.final_output)
```

For deterministic checks (token budget, span shape, expected tool sequence) you can also walk the `AgentTrace` directly without an LLM. See [docs/evaluation.md](docs/evaluation.md) for the full guide and tradeoffs.

## Serve as a service

```python
from tinyagent.serving import MCPServingConfig

agent = TinyAgent.create(...)
handle = await agent.serve_async(MCPServingConfig(port=8080))
```

A2A serving is available with the `[a2a]` extra.

## Running in Jupyter

```python
import nest_asyncio
nest_asyncio.apply()
```

## License

Apache 2.0.
