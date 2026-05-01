---
title: Config
description: AgentConfig, MCP configurations, and serving configs
---

## AgentConfig

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `model_id` | `str` | Select the underlying model used by the agent. Refer to [AnyLLM Provider Docs](https://mozilla-ai.github.io/any-llm/providers/) for the list of providers and how to access them. |
| `api_base` | `str \| None` | Custom API endpoint URL for the model provider. Use this to specify custom endpoints for local models (Ollama, llama.cpp, etc.) or proxy services. For example: `http://localhost:11434/v1` for Ollama. |
| `api_key` | `str \| None` | API key for authenticating with the model provider. By default, any-llm automatically searches for common environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.). Only set this explicitly when using custom environment variable names or providing keys dynamically. |
| `description` | `str \| None` | Description of the agent. |
| `name` | `str` | The name of the agent. Defaults to `tinyagent`. |
| `instructions` | `str \| None` | Specify the instructions for the agent (often also referred to as a `system_prompt`). |
| `tools` | `list[str \| MCPStdio \| MCPSse \| MCPStreamableHttp \| Callable[..., Any]]` | List of tools to be used by the agent. |
| `callbacks` | `list[Callback]` | List of callbacks to use during agent invocation. |
| `model_args` | `MutableMapping[str, Any] \| None` | Pass arguments to the model instance like `temperature`, `top_k`, as well as any other provider-specific parameters. Refer to [any-llm Completion API Docs](https://mozilla-ai.github.io/any-llm/api/completion/) for more info. |
| `any_llm_args` | `MutableMapping[str, Any] \| None` | Pass arguments to `AnyLLM.create()` for provider/client initialization. Use this for provider/client initialization options that are not completion-time generation params (which should be passed via `model_args`). |
| `output_type` | `type[BaseModel] \| None` | Control the output schema from calling `run`. By default, the agent will return a type str. Using this parameter you can define a Pydantic model that will be returned by the agent run methods. |

---

## MCPStdio

Configuration for running an MCP server as a local subprocess.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `command` | `str` | The executable to run to start the server. For example, `docker`, `uvx`, `npx`. |
| `args` | `Sequence[str]` |  |
| `env` | `dict[str, str] \| None` | The environment variables to set for the server. |
| `tools` | `Sequence[str] \| None` |  |
| `client_session_timeout_seconds` | `float \| None` | the read timeout passed to the MCP ClientSession. |

---

## MCPStreamableHttp

Configuration for connecting to an MCP server via Streamable HTTP transport.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `url` | `str` | The URL of the server. |
| `headers` | `Mapping[str, str] \| None` | The headers to send to the server. |
| `tools` | `Sequence[str] \| None` |  |
| `client_session_timeout_seconds` | `float \| None` | the read timeout passed to the MCP ClientSession. |

---

## MCPSse

Configuration for connecting to an MCP server via SSE transport (deprecated).

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `url` | `str` | The URL of the server. |
| `headers` | `Mapping[str, str] \| None` | The headers to send to the server. |
| `tools` | `Sequence[str] \| None` |  |
| `client_session_timeout_seconds` | `float \| None` | the read timeout passed to the MCP ClientSession. |

---

## ServingConfig

Configuration for serving an agent using the Agent2Agent Protocol (A2A).

We use the example `A2ASever` from https://github.com/google/A2A/tree/main/samples/python.

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `host` | `str` | Will be passed as argument to `uvicorn.run`. |
| `port` | `int` | Will be passed as argument to `uvicorn.run`. |
| `endpoint` | `str` | Will be pass as argument to `Starlette().add_route` |
| `log_level` | `str` | Will be passed as argument to the `uvicorn` server. |
| `version` | `str` |  |

---

## A2AServingConfig

Configuration for serving agents via the Agent2Agent Protocol.


---

## MCPServingConfig

Configuration for serving an agent using the Model Context Protocol (MCP).

**Example:**

```python
config = MCPServingConfig(
    port=8080,
    endpoint="/my-agent",
)
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `host` | `str` | Will be passed as argument to `uvicorn.run`. |
| `port` | `int` | Will be passed as argument to `uvicorn.run`. |
| `endpoint` | `str` | Will be pass as argument to `Starlette().add_route` |
| `log_level` | `str` | Will be passed as argument to the `uvicorn` server. |
| `version` | `str` |  |
