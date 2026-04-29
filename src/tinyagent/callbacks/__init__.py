from .base import Callback
from .context import Context
from .span_print import ConsolePrintSpan

__all__ = ["Callback", "ConsolePrintSpan", "Context"]


def get_default_callbacks() -> list[Callback]:
    """Return instances of the default callbacks used in tinyagent.

    This function is called internally when the user doesn't provide a
    value for [`AgentConfig.callbacks`][tinyagent.config.AgentConfig.callbacks].

    Returns:
        A list of instances containing:

            - [`ConsolePrintSpan`][tinyagent.callbacks.span_print.ConsolePrintSpan]

    """
    return [ConsolePrintSpan()]
