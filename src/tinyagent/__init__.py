from importlib.metadata import PackageNotFoundError, version

from .agent import AgentCancel, AgentRunError, TinyAgent
from .config import AgentConfig
from .tracing.agent_trace import AgentTrace

try:
    __version__ = version("mozilla-ai-tinyagent")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"

__all__ = [
    "AgentCancel",
    "AgentConfig",
    "AgentRunError",
    "AgentTrace",
    "TinyAgent",
    "__version__",
]
