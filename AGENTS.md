# AGENTS.md

This file provides guidance to coding agents (Claude Code, Cursor, Codex, etc.) working with this repository.

`CLAUDE.md` is a symlink to this file. Always edit `AGENTS.md` directly; never modify `CLAUDE.md`.

## Where to Look First

- [README.md](README.md): high-level usage and value proposition.
- [CONTRIBUTING.md](CONTRIBUTING.md): canonical dev setup and contribution workflow.
- [pyproject.toml](pyproject.toml) and [.pre-commit-config.yaml](.pre-commit-config.yaml): formatting/lint/typecheck configuration.
- [docs/](docs/): user-facing docs (mkdocs).

## Project Structure

- `src/tinyagent/`: Core library. The package is published to PyPI as `mozilla-ai-tinyagent` and imported as `tinyagent`.
- `src/tinyagent/agent.py`: The full agent loop. `TinyAgent` is the entry point; `AgentCancel` and `AgentRunError` are also defined here.
- `src/tinyagent/config.py`: `AgentConfig`, `MCPStdio`, `MCPSse`, `MCPStreamableHttp`, `ServingConfig`.
- `src/tinyagent/callbacks/`: Callback system. `Callback` (base.py), `Context` (context.py), `ConsolePrintSpan` (span_print.py), `SpanEndCallback` (span_end.py), the OTel-emitting `_SpanGeneration` (span_generation.py), and the agent-monkey-patching `_TinyAgentWrapper` (wrapper.py).
- `src/tinyagent/tools/`: Built-in tool helpers (web_browsing, user_interaction, final_output), the MCP client (`mcp/`), the optional A2A adapter (`a2a.py`), and the optional Composio adapter (`composio.py`). `wrappers.py` is the entry point for normalizing tools (callables and `MCP*` configs) into the loop.
- `src/tinyagent/tracing/`: OpenTelemetry-based tracing. `AgentTrace` and `AgentSpan` are the primary public types; `attributes.py` defines the `gen_ai.*` attribute keys.
- `src/tinyagent/serving/`: MCP and A2A server implementations for serving an agent over a network.
- `src/tinyagent/evaluation/`: `LlmJudge` and `AgentJudge` evaluators.
- `src/tinyagent/testing/helpers.py`: Helpers used by tests (and consumable by downstream test suites).
- `tests/`: `unit/`, `integration/`, `snapshots/`, plus shared fixtures in `tests/conftest.py`.

## Build, Test, and Development Commands

This repo uses `uv` for local dev (Python 3.11+).

- Create env + install dev deps: `uv venv && source .venv/bin/activate && uv pip install -e '.[all]' && uv sync --group tests`
- Run all checks: `uv run pre-commit run --all-files --verbose`
- Unit tests: `uv run pytest tests/unit tests/snapshots`
- Integration tests (require `MISTRAL_API_KEY` and/or `OPENAI_API_KEY`): `uv run pytest tests/integration`

## Coding Style & Naming Conventions

- Python indentation: 4 spaces; formatting/linting via `ruff` and `pre-commit`. Line length is not explicitly capped (E501 is ignored).
- Type hints: required; `mypy` runs in strict mode for library code (see `pyproject.toml`).
- The `tinyagent` package is intentionally small. Resist adding new abstractions; the value proposition is simplicity.
- Lazy imports are used for optional dependencies (`a2a-sdk`, `composio`). The ruff rule `PLC0415` is disabled to allow this.
- Prefer direct attribute access (e.g., `obj.field`) over `getattr(obj, "field")` when the field is typed.
- Add code comments only when the *why* is non-obvious. Never use em-dashes or `--` in comments or descriptions.

## Testing Guidelines

- Framework: `pytest` (+ `pytest-asyncio`, `pytest-xdist`). Async mode is `auto` with session-scoped event loop.
- Add/adjust tests with every change (happy path + error cases). Integration tests should skip cleanly via `pytest.skip(...)` or `@pytest.mark.skipif` when credentials are missing.
- Do not use class-based test grouping (`class TestFoo:`). All tests are standalone functions.
- Do not add decorative section-separator comments.
- Place imports at the top of test files unless the import is for an optional dependency that may not be installed.
- Snapshot tests live in `tests/snapshots/` using `syrupy`.

## Architecture Notes

- `TinyAgent.create(config)` / `create_async(config)` is the main factory. It instantiates the agent and calls `_load_agent()` (which connects MCP clients and builds the OpenAI-style tools list).
- The agent loop is in `TinyAgent._run_async`. It calls the LLM in a loop, dispatches tool calls (including a `final_answer` tool), and returns when the LLM emits a final answer or content message.
- `_TinyAgentWrapper` (in `callbacks/wrapper.py`) monkey-patches `agent.call_model` and per-tool `call_tool` so that user callbacks fire `before_*` and `after_*` for each LLM call and tool execution.
- `_SpanGeneration` (in `callbacks/span_generation.py`) is registered automatically and converts each LLM call and tool execution into an OTel span on the run's `AgentTrace`.
- `AgentCancel` is a control-flow exception. Subclass it in user callbacks when you want to *intentionally* stop a run; it propagates without being wrapped in `AgentRunError`.
- `AgentRunError` wraps any other exception raised during a run and preserves the partial `AgentTrace`.
- The `any-llm-sdk` package provides the unified LLM interface (`AnyLLM.create()` for clients, `acompletion()` for the call). `model_id` follows the `provider:model` syntax from any-llm.

## Relationship to `any-agent`

This package was extracted from the multi-framework [`any-agent`](https://github.com/mozilla-ai/any-agent) library. `any-agent` now depends on `mozilla-ai-tinyagent` and re-exports the shared callback/tracing types so that both packages share class identities at runtime. When making changes here, prefer changes that don't break that re-export contract.

## Commit & Pull Request Guidelines

- Commits follow Conventional Commits: `feat(scope): ...`, `fix: ...`, `chore(deps): ...`, `tests: ...`.
- PRs should include a clear description and linked issues.

## Security & Configuration Tips

- Never commit secrets. Use environment variables or a local `.env` (gitignored) for provider API keys.
