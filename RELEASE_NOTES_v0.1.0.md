# v0.1.0 — Initial release

`tinyagent` (PyPI: `mozilla-ai-tinyagent`) is a small, dependency-light Python implementation of the [HuggingFace Tiny Agents](https://huggingface.co/blog/tiny-agents) loop, built on [`any-llm`](https://github.com/mozilla-ai/any-llm). It was extracted from [`any-agent`](https://github.com/mozilla-ai/any-agent) into its own package so it can ship and version independently.

## Highlights

- **One readable agent loop.** `TinyAgent._run_async` in `src/tinyagent/agent.py` is the whole thing: call the LLM, dispatch tool calls (including a `final_answer` tool), repeat until done.
- **First-class tracing.** Every run returns an `AgentTrace` with OpenTelemetry-compatible spans, token counts, and cost. Spans follow the GenAI semantic conventions.
- **Callbacks for everything.** Subclass `Callback` for guardrails, metrics, observability, or intentional cancellation via `AgentCancel`. `_TinyAgentWrapper` monkey-patches `call_model` and per-tool `call_tool` so hooks fire around every LLM call and tool execution.
- **MCP-native.** First-class support for stdio, SSE, and streamable HTTP servers via `MCPStdio`, `MCPSse`, `MCPStreamableHttp`.
- **Provider-agnostic.** Built on `any-llm-sdk`. Swap providers by changing `model_id` (`mistral:mistral-small-latest`, `openai:gpt-4o-mini`, etc.).
- **Serve as a service.** Run your agent over MCP or A2A with `agent.serve_async(...)`.
- **Evaluation utilities.** `LlmJudge` and `AgentJudge` for scoring runs.

## Install

```bash
pip install mozilla-ai-tinyagent
```

Distribution name is `mozilla-ai-tinyagent`; import name is `tinyagent`.

Optional extras:

```bash
pip install 'mozilla-ai-tinyagent[a2a]'       # A2A serving
pip install 'mozilla-ai-tinyagent[composio]'  # Composio tools
pip install 'mozilla-ai-tinyagent[all]'       # everything
```

Requires Python 3.11–3.13.

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

## Relationship to `any-agent`

`any-agent` now depends on `mozilla-ai-tinyagent` and re-exports the shared callback / tracing types so that both packages share class identities at runtime. Existing `AnyAgent.create(AgentFramework.TINYAGENT, ...)` users are unaffected.

## Docs

- Full docs: https://docs.mozilla.ai/tinyagent/
- Cookbook: https://docs.mozilla.ai/tinyagent/cookbook/your-first-agent/
- Contributing: see [CONTRIBUTING.md](CONTRIBUTING.md)
