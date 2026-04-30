# tinyagent

A minimal agent framework with first-class tracing, callbacks, MCP, and serving support.

## Why tinyagent

- **Tiny**: One small Python implementation of the agent loop. You can read it in one sitting.
- **Tracing built in**: Every run produces an `AgentTrace` with OpenTelemetry-compatible spans, token counts, and cost.
- **Callbacks**: Subclass `Callback` for guardrails, metrics, observability, or intentional cancellation.
- **MCP-native**: First-class support for [Model Context Protocol](https://modelcontextprotocol.io/) tools (stdio, SSE, streamable HTTP).
- **Serve as a service**: Run your agent over MCP or A2A.
- **Provider-agnostic**: Built on [`any-llm`](https://github.com/mozilla-ai/any-llm). Swap providers by changing `model_id`.

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

## What's next

- [Models](models.md) — configuring providers and `model_id` syntax.
- [Tools](tools.md) — using callable tools, MCP servers, and the built-in helpers.
- [Callbacks](callbacks.md) — observe and control execution.
- [Tracing](tracing.md) — inspect spans, tokens, and cost.
- [Serving](serving.md) — serve your agent over MCP or A2A.
- [Evaluation](evaluation.md) — judge an agent's behavior with `LlmJudge` and `AgentJudge`.
- [Cookbook](cookbook/your-first-agent.md) — runnable examples.
