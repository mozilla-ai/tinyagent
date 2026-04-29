import warnings
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tinyagent.callbacks import get_default_callbacks
from tinyagent.callbacks.base import Callback


class MCPStdio(BaseModel):
    command: str
    """The executable to run to start the server.

    For example, `docker`, `uvx`, `npx`.
    """

    args: Sequence[str]
    """Command line args to pass to the command executable.

    For example, `["run", "-i", "--rm", "mcp/fetch"]`.
    """

    env: dict[str, str] | None = None
    """The environment variables to set for the server."""

    tools: Sequence[str] | None = None
    """List of tool names to use from the MCP Server.

    Use it to limit the tools accessible by the agent.
    For example, if you use [`mcp/filesystem`](https://hub.docker.com/r/mcp/filesystem),
    you can pass `tools=["read_file", "list_directory"]` to limit the agent to read-only operations.

    If none is specified, the default behavior is that the agent will have access to all tools under that MCP server.
    """

    client_session_timeout_seconds: float | None = 5
    """the read timeout passed to the MCP ClientSession."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class MCPSse(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def sse_deprecation(cls, data: Any) -> Any:
        warnings.warn(
            "SSE is deprecated in the MCP specification in favor of Streamable HTTP as of version 2025-03-26",
            DeprecationWarning,
            stacklevel=2,
        )
        return data

    url: str
    """The URL of the server."""

    headers: Mapping[str, str] | None = None
    """The headers to send to the server."""

    tools: Sequence[str] | None = None
    """List of tool names to use from the MCP Server.

    Use it to limit the tools accessible by the agent.
    For example, if you use [`mcp/filesystem`](https://hub.docker.com/r/mcp/filesystem),
    you can pass `tools=["read_file", "list_directory"]` to limit the agent to read-only operations.
    """

    client_session_timeout_seconds: float | None = 5
    """the read timeout passed to the MCP ClientSession."""

    model_config = ConfigDict(frozen=True)


class MCPStreamableHttp(BaseModel):
    url: str
    """The URL of the server."""

    headers: Mapping[str, str] | None = None
    """The headers to send to the server."""

    tools: Sequence[str] | None = None
    """List of tool names to use from the MCP Server.

    Use it to limit the tools accessible by the agent.
    For example, if you use [`mcp/filesystem`](https://hub.docker.com/r/mcp/filesystem),
    you can pass `tools=["read_file", "list_directory"]` to limit the agent to read-only operations.
    """

    client_session_timeout_seconds: float | None = 5
    """the read timeout passed to the MCP ClientSession."""

    model_config = ConfigDict(frozen=True)


class ServingConfig(BaseModel):
    """Configuration for serving an agent using the Agent2Agent Protocol (A2A).

    We use the example `A2ASever` from https://github.com/google/A2A/tree/main/samples/python.
    """

    model_config = ConfigDict(extra="forbid")

    host: str = "localhost"
    """Will be passed as argument to `uvicorn.run`."""

    port: int = 5000
    """Will be passed as argument to `uvicorn.run`."""

    endpoint: str = "/"
    """Will be pass as argument to `Starlette().add_route`"""

    log_level: str = "warning"
    """Will be passed as argument to the `uvicorn` server."""

    version: str = "0.1.0"


MCPParams = MCPStdio | MCPSse | MCPStreamableHttp

Tool = str | MCPParams | Callable[..., Any]


class AgentConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model_id: str
    """Select the underlying model used by the agent.

    Refer to [AnyLLM Provider Docs](https://mozilla-ai.github.io/any-llm/providers/)
    for the list of providers and how to access them.
    """

    api_base: str | None = None
    """Custom API endpoint URL for the model provider.

    Use this to specify custom endpoints for local models (Ollama, llama.cpp, etc.) or proxy services.
    For example: `http://localhost:11434/v1` for Ollama.
    """

    api_key: str | None = None
    """API key for authenticating with the model provider.

    By default, any-llm automatically searches for common environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.).
    Only set this explicitly when using custom environment variable names or providing keys dynamically.
    """

    description: str | None = None
    """Description of the agent."""

    name: str = "tinyagent"
    """The name of the agent.

    Defaults to `tinyagent`.
    """

    instructions: str | None = None
    """Specify the instructions for the agent (often also referred to as a `system_prompt`)."""

    tools: list[Tool] = Field(default_factory=list)
    """List of tools to be used by the agent."""

    callbacks: list[Callback] = Field(default_factory=get_default_callbacks)
    """List of callbacks to use during agent invocation."""

    model_args: MutableMapping[str, Any] | None = None
    """Pass arguments to the model instance like `temperature`, `top_k`, as well as any other provider-specific parameters.

    Refer to [any-llm Completion API Docs](https://mozilla-ai.github.io/any-llm/api/completion/) for more info.
    """

    any_llm_args: MutableMapping[str, Any] | None = None
    """Pass arguments to `AnyLLM.create()` for provider/client initialization.

    Use this for provider/client initialization options that are not completion-time
    generation params (which should be passed via `model_args`).
    """

    output_type: type[BaseModel] | None = None
    """Control the output schema from calling `run`. By default, the agent will return a type str.

    Using this parameter you can define a Pydantic model that will be returned by the agent run methods.
    """
