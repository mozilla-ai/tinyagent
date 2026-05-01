"""Generate API reference documentation from Python source code.

Extracts docstrings and signatures from the tinyagent source and generates
markdown pages. Generated files are written to docs/api/ and should not be
committed to git.

Usage:
    python scripts/generate_api_docs.py
"""

from __future__ import annotations

import inspect
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, get_type_hints

DOCS_API_DIR = Path(__file__).parent.parent / "docs" / "api"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _short_name(name: str) -> str:
    """Strip module prefixes to produce a short readable type name."""
    prefixes = [
        "tinyagent.tracing.agent_trace.",
        "tinyagent.tracing.attributes.",
        "tinyagent.callbacks.base.",
        "tinyagent.callbacks.context.",
        "tinyagent.callbacks.span_print.",
        "tinyagent.evaluation.",
        "tinyagent.serving.",
        "tinyagent.config.",
        "tinyagent.tools.",
        "tinyagent.agent.",
        "tinyagent.",
        "collections.abc.",
        "pydantic.main.",
        "pydantic.",
        "typing.",
        "typing_extensions.",
        "opentelemetry.trace.",
        "opentelemetry.sdk.trace.",
    ]
    for prefix in prefixes:
        name = name.removeprefix(prefix)
    return name


def _clean_qualified_names(text: str) -> str:
    """Remove module prefixes from qualified names in generated text."""
    replacements = [
        ("tinyagent.tracing.agent_trace.", ""),
        ("tinyagent.tracing.attributes.", ""),
        ("tinyagent.callbacks.base.", ""),
        ("tinyagent.callbacks.context.", ""),
        ("tinyagent.callbacks.span_print.", ""),
        ("tinyagent.evaluation.", ""),
        ("tinyagent.serving.", ""),
        ("tinyagent.config.", ""),
        ("tinyagent.tools.", ""),
        ("tinyagent.agent.", ""),
        ("tinyagent.", ""),
        ("collections.abc.", ""),
        ("pydantic.main.", ""),
        ("pydantic.", ""),
        ("typing.", ""),
        ("typing_extensions.", ""),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return re.sub(r"\[(`[^`]+`)\]\[[^\]]+\]", r"\1", text)


def _format_annotation(annotation: Any) -> str:
    if annotation is inspect.Parameter.empty:
        return ""
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", None)

    if origin is not None and args is not None:
        origin_name = getattr(origin, "__name__", str(origin))
        origin_name = _short_name(origin_name)
        if origin_name == "Union" or "Union" in str(origin):
            parts = [_format_annotation(a) for a in args]
            parts = ["None" if p == "NoneType" else p for p in parts]
            return " | ".join(parts)
        if origin_name == "Literal" or "Literal" in str(origin):
            formatted_args = ", ".join(
                repr(a) if isinstance(a, str) else str(a) for a in args
            )
            return f"Literal[{formatted_args}]"
        formatted_args = ", ".join(_format_annotation(a) for a in args)
        return f"{origin_name}[{formatted_args}]"

    if hasattr(annotation, "__name__"):
        return _short_name(annotation.__name__)
    return _short_name(str(annotation))


def _format_default(default: Any) -> str:
    if default is inspect.Parameter.empty:
        return ""
    if default is None:
        return "None"
    if isinstance(default, str):
        return f'"{default}"'
    return str(default)


def _get_signature_block(func: Any, func_name: str | None = None) -> str:
    name = func_name or func.__name__
    is_async = inspect.iscoroutinefunction(func)
    prefix = "async " if is_async else ""

    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return f"```python\n{prefix}def {name}(...)\n```"

    params = list(sig.parameters.values())
    lines = []
    seen_kw_only = False
    for p in params:
        if p.kind == inspect.Parameter.KEYWORD_ONLY and not seen_kw_only:
            seen_kw_only = True
            if lines:
                lines.append("    *,")
        ann = (
            _format_annotation(p.annotation)
            if p.annotation is not inspect.Parameter.empty
            else ""
        )
        default = _format_default(p.default)
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            part = f"**{p.name}"
            if ann:
                part += f": {ann}"
        elif p.kind == inspect.Parameter.VAR_POSITIONAL:
            part = f"*{p.name}"
            if ann:
                part += f": {ann}"
        else:
            part = p.name
            if ann:
                part += f": {ann}"
            if default:
                part += f" = {default}"
        lines.append(f"    {part},")

    ret = ""
    if sig.return_annotation is not inspect.Signature.empty:
        ret = f" -> {_format_annotation(sig.return_annotation)}"

    if not lines:
        result = f"```python\n{prefix}def {name}(){ret}\n```"
    else:
        body = "\n".join(lines)
        result = f"```python\n{prefix}def {name}(\n{body}\n){ret}\n```"

    return _clean_qualified_names(result)


def _join_summary_lines(lines: list[str]) -> str:
    paragraphs: list[list[str]] = [[]]
    for line in lines:
        if line == "":
            paragraphs.append([])
        else:
            paragraphs[-1].append(line)

    def _render_paragraph(p: list[str]) -> str:
        if any(line.startswith(("- ", "* ")) for line in p):
            return "\n".join(p)
        return " ".join(p)

    return "\n\n".join(_render_paragraph(p) for p in paragraphs if p).strip()


def _parse_docstring(docstring: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {"summary": "", "args": {}, "returns": "", "raises": []}
    if not docstring:
        return result

    dedented = textwrap.dedent(docstring).strip()
    lines = dedented.split("\n")
    current_section = "summary"
    current_arg = ""
    summary_lines: list[str] = []
    example_lines: list[str] = []
    example_indent: int = 0

    for line in lines:
        stripped = line.strip()

        if stripped in ("Args:", "Attributes:"):
            current_section = "args"
            continue
        if stripped == "Returns:":
            current_section = "returns"
            continue
        if stripped == "Raises:":
            current_section = "raises"
            continue
        if re.match(r"^Examples?:$", stripped):
            current_section = "example"
            continue

        if current_section == "summary":
            summary_lines.append(stripped)
        elif current_section == "example":
            if example_lines or stripped:
                if not example_lines and stripped:
                    example_indent = len(line) - len(line.lstrip())
                example_lines.append(line[example_indent:])
        elif current_section == "args":
            m = re.match(r"^(\*{0,2}\w+)\s*(?:\(.*?\))?\s*:\s*(.*)", stripped)
            if m:
                current_arg = m.group(1)
                result["args"][current_arg] = m.group(2).strip()
            elif current_arg and stripped:
                result["args"][current_arg] += " " + stripped
        elif current_section == "returns":
            if stripped:
                sep = (
                    "\n"
                    if stripped.startswith(("- ", "* "))
                    else (" " if result["returns"] else "")
                )
                result["returns"] += sep + stripped
        elif current_section == "raises":
            m = re.match(r"^(\w+)\s*:\s*(.*)", stripped)
            if m:
                result["raises"].append(
                    {"type": m.group(1), "desc": m.group(2).strip()}
                )
            elif result["raises"] and stripped:
                result["raises"][-1]["desc"] += " " + stripped

    summary = _join_summary_lines(summary_lines)
    if example_lines:
        code = "\n".join(example_lines).strip()
        code = re.sub(r"^```[a-z]*\n?", "", code)
        code = re.sub(r"\n?```$", "", code).strip()
        summary = (
            f"{summary}\n\n**Example:**\n\n```python\n{code}\n```"
            if summary
            else f"**Example:**\n\n```python\n{code}\n```"
        )
    result["summary"] = _clean_qualified_names(summary)
    result["returns"] = _clean_qualified_names(result["returns"])
    return result


def _param_table(func: Any, parsed_doc: dict[str, Any]) -> str:
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        return ""

    rows = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        ann = (
            _format_annotation(param.annotation)
            if param.annotation is not inspect.Parameter.empty
            else ""
        )
        default = _format_default(param.default)
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            display_name = f"**{name}"
        elif param.kind == inspect.Parameter.VAR_POSITIONAL:
            display_name = f"*{name}"
        else:
            display_name = name

        doc_desc = parsed_doc["args"].get(name, parsed_doc["args"].get(f"**{name}", ""))
        ann_escaped = ann.replace("|", "\\|")
        if not default and param.default is inspect.Parameter.empty:
            default = "*required*"
        elif not default:
            default = ""
        rows.append((display_name, ann_escaped, default, doc_desc))

    if not rows:
        return ""

    has_descriptions = any(r[3] for r in rows)
    if has_descriptions:
        header = "| Parameter | Type | Default | Description |\n|-----------|------|---------|-------------|"
        table_rows = [f"| `{r[0]}` | `{r[1]}` | {r[2]} | {r[3]} |" for r in rows]
    else:
        header = "| Parameter | Type | Default |\n|-----------|------|---------|"
        table_rows = [f"| `{r[0]}` | `{r[1]}` | {r[2]} |" for r in rows]
    return _clean_qualified_names(header + "\n" + "\n".join(table_rows))


def _pydantic_field_table(cls: type) -> str:
    rows = []
    hints = {}
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = getattr(cls, "__annotations__", {})

    model_fields = getattr(cls, "model_fields", None)

    for field_name, field_type in hints.items():
        if field_name.startswith("_") or field_name == "model_config":
            continue

        ann = _format_annotation(field_type).replace("|", "\\|")
        desc = ""
        if model_fields and field_name in model_fields:
            fi = model_fields[field_name]
            desc = fi.description or ""

        if not desc:
            try:
                source = inspect.getsource(cls)
                pattern = rf'(?:^|\n)\s+{re.escape(field_name)}\s*[:=][^\n]*\n\s+"""([^"]*(?:""[^"])*?)"""'
                m = re.search(pattern, source)
                if m:
                    desc = " ".join(m.group(1).strip().split())
            except (OSError, TypeError):
                pass

        rows.append(f"| `{field_name}` | `{ann}` | {desc} |")

    if not rows:
        return ""

    header = "| Field | Type | Description |\n|-------|------|-------------|"
    return _clean_qualified_names(header + "\n" + "\n".join(rows))


def _generate_function_doc(
    func: Any,
    heading_level: str = "##",
    module_prefix: str = "tinyagent",
    func_name: str | None = None,
) -> str:
    name = func_name or func.__name__
    parsed = _parse_docstring(func.__doc__)
    parts = [f"{heading_level} `{module_prefix}.{name}()`"]
    if parsed["summary"]:
        parts.append("")
        parts.append(parsed["summary"])
    parts.append("")
    parts.append(_get_signature_block(func, name))
    table = _param_table(func, parsed)
    if table:
        parts.append("")
        parts.append(f"{heading_level}# Parameters")
        parts.append("")
        parts.append(table)
    if parsed["returns"]:
        parts.append("")
        parts.append(f"**Returns:** {parsed['returns']}")
    if parsed["raises"]:
        parts.append("")
        parts.append("**Raises:**")
        for r in parsed["raises"]:
            parts.append(f"- `{r['type']}`: {r['desc']}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Page generators
# ---------------------------------------------------------------------------


def generate_agent_page() -> str:
    from tinyagent import AgentCancel, AgentRunError, TinyAgent

    parts = [
        "---",
        "title: Agent",
        "description: TinyAgent, AgentCancel, and AgentRunError reference",
        "---",
        "",
    ]

    # TinyAgent
    parts.append("## TinyAgent")
    parts.append("")
    parsed = _parse_docstring(TinyAgent.__doc__)
    if parsed["summary"]:
        parts.append(parsed["summary"])
        parts.append("")

    parts.append("### `TinyAgent.create()`")
    parts.append("")
    create_parsed = _parse_docstring(TinyAgent.create.__doc__)
    if create_parsed["summary"]:
        parts.append(create_parsed["summary"])
        parts.append("")
    parts.append(_get_signature_block(TinyAgent.create, "create"))
    table = _param_table(TinyAgent.create, create_parsed)
    if table:
        parts.append("")
        parts.append(table)
    parts.append("")

    parts.append("### `TinyAgent.create_async()`")
    parts.append("")
    parts.append("Async variant of `create()` with the same parameters.")
    parts.append("")
    parts.append(_get_signature_block(TinyAgent.create_async, "create_async"))
    parts.append("")

    parts.append("### `TinyAgent.run()`")
    parts.append("")
    run_parsed = _parse_docstring(TinyAgent.run.__doc__)
    if run_parsed["summary"]:
        parts.append(run_parsed["summary"])
        parts.append("")
    parts.append(_get_signature_block(TinyAgent.run, "run"))
    table = _param_table(TinyAgent.run, run_parsed)
    if table:
        parts.append("")
        parts.append(table)
    if run_parsed["returns"]:
        parts.append("")
        parts.append(f"**Returns:** {run_parsed['returns']}")
    parts.append("")

    parts.append("### `TinyAgent.run_async()`")
    parts.append("")
    parts.append("Async variant of `run()` with the same parameters.")
    parts.append("")
    parts.append(_get_signature_block(TinyAgent.run_async, "run_async"))
    parts.append("")

    parts.append("### `TinyAgent.serve_async()`")
    parts.append("")
    serve_parsed = _parse_docstring(TinyAgent.serve_async.__doc__)
    if serve_parsed["summary"]:
        parts.append(serve_parsed["summary"])
        parts.append("")
    parts.append(_get_signature_block(TinyAgent.serve_async, "serve_async"))
    parts.append("")

    parts.append("### `TinyAgent.cleanup_async()`")
    parts.append("")
    parts.append(
        "Clean up resources (MCP connections, etc.). Called automatically when using the async context manager pattern."
    )
    parts.append("")

    # AgentCancel
    parts.append("---")
    parts.append("")
    parts.append("## AgentCancel")
    parts.append("")
    parsed = _parse_docstring(AgentCancel.__doc__)
    if parsed["summary"]:
        parts.append(parsed["summary"])
        parts.append("")
    parts.append("**Properties:**")
    parts.append("")
    parts.append(
        "- `trace` - `AgentTrace | None`: Execution trace collected before cancellation."
    )
    parts.append("")

    # AgentRunError
    parts.append("---")
    parts.append("")
    parts.append("## AgentRunError")
    parts.append("")
    parsed = _parse_docstring(AgentRunError.__doc__)
    if parsed["summary"]:
        parts.append(parsed["summary"])
        parts.append("")
    parts.append("**Properties:**")
    parts.append("")
    parts.append(
        "- `trace` - `AgentTrace`: The execution trace collected up to failure point."
    )
    parts.append(
        "- `original_exception` - `Exception`: The underlying exception that was caught."
    )
    parts.append("")

    return "\n".join(parts)


def generate_config_page() -> str:
    from tinyagent.config import (
        AgentConfig,
        MCPSse,
        MCPStdio,
        MCPStreamableHttp,
        ServingConfig,
    )

    parts = [
        "---",
        "title: Config",
        "description: AgentConfig, MCP configurations, and serving configs",
        "---",
        "",
    ]

    # AgentConfig
    parts.append("## AgentConfig")
    parts.append("")
    parsed = _parse_docstring(AgentConfig.__doc__)
    if parsed["summary"]:
        parts.append(parsed["summary"])
        parts.append("")
    table = _pydantic_field_table(AgentConfig)
    if table:
        parts.append("### Fields")
        parts.append("")
        parts.append(table)
    parts.append("")

    # MCPStdio
    parts.append("---")
    parts.append("")
    parts.append("## MCPStdio")
    parts.append("")
    parsed = _parse_docstring(MCPStdio.__doc__)
    if parsed["summary"]:
        parts.append(parsed["summary"])
        parts.append("")
    else:
        parts.append("Configuration for running an MCP server as a local subprocess.")
        parts.append("")
    table = _pydantic_field_table(MCPStdio)
    if table:
        parts.append("### Fields")
        parts.append("")
        parts.append(table)
    parts.append("")

    # MCPStreamableHttp
    parts.append("---")
    parts.append("")
    parts.append("## MCPStreamableHttp")
    parts.append("")
    parsed = _parse_docstring(MCPStreamableHttp.__doc__)
    if parsed["summary"]:
        parts.append(parsed["summary"])
        parts.append("")
    else:
        parts.append(
            "Configuration for connecting to an MCP server via Streamable HTTP transport."
        )
        parts.append("")
    table = _pydantic_field_table(MCPStreamableHttp)
    if table:
        parts.append("### Fields")
        parts.append("")
        parts.append(table)
    parts.append("")

    # MCPSse
    parts.append("---")
    parts.append("")
    parts.append("## MCPSse")
    parts.append("")
    parsed = _parse_docstring(MCPSse.__doc__)
    if parsed["summary"]:
        parts.append(parsed["summary"])
        parts.append("")
    else:
        parts.append(
            "Configuration for connecting to an MCP server via SSE transport (deprecated)."
        )
        parts.append("")
    table = _pydantic_field_table(MCPSse)
    if table:
        parts.append("### Fields")
        parts.append("")
        parts.append(table)
    parts.append("")

    # ServingConfig (base / A2A)
    parts.append("---")
    parts.append("")
    parts.append("## ServingConfig")
    parts.append("")
    parsed = _parse_docstring(ServingConfig.__doc__)
    if parsed["summary"]:
        parts.append(parsed["summary"])
        parts.append("")
    table = _pydantic_field_table(ServingConfig)
    if table:
        parts.append("### Fields")
        parts.append("")
        parts.append(table)
    parts.append("")

    # A2AServingConfig / MCPServingConfig
    try:
        from tinyagent.serving import A2AServingConfig, MCPServingConfig

        parts.append("---")
        parts.append("")
        parts.append("## A2AServingConfig")
        parts.append("")
        parsed = _parse_docstring(A2AServingConfig.__doc__)
        if parsed["summary"]:
            parts.append(parsed["summary"])
            parts.append("")
        else:
            parts.append(
                "Configuration for serving agents via the Agent2Agent Protocol."
            )
            parts.append("")
        table = _pydantic_field_table(A2AServingConfig)
        if table:
            parts.append("### Fields")
            parts.append("")
            parts.append(table)
        parts.append("")

        parts.append("---")
        parts.append("")
        parts.append("## MCPServingConfig")
        parts.append("")
        parsed = _parse_docstring(MCPServingConfig.__doc__)
        if parsed["summary"]:
            parts.append(parsed["summary"])
            parts.append("")
        else:
            parts.append(
                "Configuration for serving agents via the Model Context Protocol."
            )
            parts.append("")
        table = _pydantic_field_table(MCPServingConfig)
        if table:
            parts.append("### Fields")
            parts.append("")
            parts.append(table)
        parts.append("")
    except ImportError:
        parts.append("---")
        parts.append("")
        parts.append("## A2AServingConfig")
        parts.append("")
        parts.append(
            "Configuration for serving agents via A2A. Install `mozilla-ai-tinyagent[a2a]` for full docs."
        )
        parts.append("")
        parts.append("## MCPServingConfig")
        parts.append("")
        parts.append("Configuration for serving agents via MCP.")
        parts.append("")

    return "\n".join(parts)


def generate_callbacks_page() -> str:
    from tinyagent.callbacks import Callback, Context, get_default_callbacks

    parts = [
        "---",
        "title: Callbacks",
        "description: Callback, Context, ConsolePrintSpan, and get_default_callbacks reference",
        "---",
        "",
    ]

    # Callback
    parts.append("## Callback")
    parts.append("")
    parsed = _parse_docstring(Callback.__doc__)
    if parsed["summary"]:
        parts.append(parsed["summary"])
        parts.append("")
    parts.append(
        "Base class for tinyagent callbacks. Subclass and override any subset of the lifecycle methods."
    )
    parts.append("")

    for method_name in [
        "before_agent_invocation",
        "before_llm_call",
        "after_llm_call",
        "before_tool_execution",
        "after_tool_execution",
        "after_agent_invocation",
    ]:
        method = getattr(Callback, method_name, None)
        if method:
            parts.append(f"### `Callback.{method_name}()`")
            parts.append("")
            method_parsed = _parse_docstring(method.__doc__)
            if method_parsed["summary"]:
                parts.append(method_parsed["summary"])
                parts.append("")
            parts.append(_get_signature_block(method, method_name))
            parts.append("")

    # Context
    parts.append("---")
    parts.append("")
    parts.append("## Context")
    parts.append("")
    parsed = _parse_docstring(Context.__doc__)
    if parsed["summary"]:
        parts.append(parsed["summary"])
        parts.append("")
    parts.append(
        "Shared context object passed through all callbacks during an agent run."
    )
    parts.append("")
    parts.append("### Fields")
    parts.append("")
    parts.append("| Field | Type | Description |")
    parts.append("|-------|------|-------------|")
    parts.append(
        "| `current_span` | `Span` | The active OpenTelemetry span with attributes (see GenAI) |"
    )
    parts.append("| `trace` | `AgentTrace` | Current execution trace |")
    parts.append("| `tracer` | `Tracer` | OpenTelemetry tracer instance |")
    parts.append(
        "| `shared` | `dict[str, Any]` | Arbitrary shared state across callbacks |"
    )
    parts.append("")

    # ConsolePrintSpan
    parts.append("---")
    parts.append("")
    parts.append("## ConsolePrintSpan")
    parts.append("")
    try:
        from tinyagent.callbacks.span_print import ConsolePrintSpan

        parsed = _parse_docstring(ConsolePrintSpan.__doc__)
        if parsed["summary"]:
            parts.append(parsed["summary"])
            parts.append("")
        else:
            parts.append(
                "Default callback that prints span information to the console."
            )
            parts.append("")
    except ImportError:
        parts.append("Default callback that prints span information to the console.")
        parts.append("")

    # get_default_callbacks
    parts.append("---")
    parts.append("")
    parts.append(
        _generate_function_doc(get_default_callbacks, "##", "tinyagent.callbacks")
    )
    parts.append("")

    return "\n".join(parts)


def generate_tracing_page() -> str:
    from tinyagent.tracing.agent_trace import AgentSpan, AgentTrace, CostInfo, TokenInfo
    from tinyagent.tracing.attributes import GenAI

    parts = [
        "---",
        "title: Tracing",
        "description: AgentTrace, AgentSpan, CostInfo, TokenInfo, and GenAI reference",
        "---",
        "",
    ]

    # AgentTrace
    parts.append("## AgentTrace")
    parts.append("")
    parsed = _parse_docstring(AgentTrace.__doc__)
    if parsed["summary"]:
        parts.append(parsed["summary"])
        parts.append("")
    table = _pydantic_field_table(AgentTrace)
    if table:
        parts.append("### Fields")
        parts.append("")
        parts.append(table)
        parts.append("")

    parts.append("### Properties")
    parts.append("")
    parts.append("- `duration` - `timedelta`: Duration of the agent invocation span.")
    parts.append("- `tokens` - `TokenInfo`: Total token usage across all LLM calls.")
    parts.append("- `cost` - `CostInfo`: Total cost across all LLM calls.")
    parts.append(
        "- `final_output` - `str | None`: The final answer returned by the agent."
    )
    parts.append("")

    # AgentSpan
    parts.append("---")
    parts.append("")
    parts.append("## AgentSpan")
    parts.append("")
    parsed = _parse_docstring(AgentSpan.__doc__)
    if parsed["summary"]:
        parts.append(parsed["summary"])
        parts.append("")
    else:
        parts.append("Serializable representation of an OpenTelemetry span.")
        parts.append("")
    table = _pydantic_field_table(AgentSpan)
    if table:
        parts.append("### Fields")
        parts.append("")
        parts.append(table)
        parts.append("")

    parts.append("### Methods")
    parts.append("")
    for method_name in [
        "is_agent_invocation",
        "is_llm_call",
        "is_tool_execution",
    ]:
        method = getattr(AgentSpan, method_name, None)
        if method:
            parts.append(f"#### `AgentSpan.{method_name}()`")
            parts.append("")
            method_parsed = _parse_docstring(method.__doc__)
            if method_parsed["summary"]:
                parts.append(method_parsed["summary"])
                parts.append("")
            parts.append(_get_signature_block(method, method_name))
            parts.append("")

    # CostInfo
    parts.append("---")
    parts.append("")
    parts.append("## CostInfo")
    parts.append("")
    table = _pydantic_field_table(CostInfo)
    if table:
        parts.append(table)
        parts.append("")
    parts.append("**Properties:** `total_cost` - `float`: Total cost (input + output).")
    parts.append("")

    # TokenInfo
    parts.append("---")
    parts.append("")
    parts.append("## TokenInfo")
    parts.append("")
    table = _pydantic_field_table(TokenInfo)
    if table:
        parts.append(table)
        parts.append("")
    parts.append(
        "**Properties:** `total_tokens` - `int`: Total tokens (input + output)."
    )
    parts.append("")

    # GenAI
    parts.append("---")
    parts.append("")
    parts.append("## GenAI")
    parts.append("")
    parts.append(
        "Constants for accessing span attributes following OpenTelemetry semantic conventions for generative AI systems."
    )
    parts.append("")
    parts.append("### Attributes")
    parts.append("")
    parts.append("| Attribute | Value |")
    parts.append("|-----------|-------|")
    for attr_name in sorted(dir(GenAI)):
        if attr_name.startswith("_"):
            continue
        attr_val = getattr(GenAI, attr_name)
        if isinstance(attr_val, str):
            parts.append(f"| `GenAI.{attr_name}` | `{attr_val}` |")
    parts.append("")

    return "\n".join(parts)


def generate_evaluation_page() -> str:
    from tinyagent.evaluation import AgentJudge, LlmJudge

    parts = [
        "---",
        "title: Evaluation",
        "description: LlmJudge and AgentJudge reference",
        "---",
        "",
    ]

    # LlmJudge
    parts.append("## LlmJudge")
    parts.append("")
    parsed = _parse_docstring(LlmJudge.__doc__)
    if parsed["summary"]:
        parts.append(parsed["summary"])
        parts.append("")
    parts.append("### Constructor")
    parts.append("")
    init_parsed = _parse_docstring(LlmJudge.__init__.__doc__)
    parts.append(_get_signature_block(LlmJudge.__init__, "__init__"))
    table = _param_table(LlmJudge.__init__, init_parsed)
    if table:
        parts.append("")
        parts.append(table)
    parts.append("")

    for method_name in ["run", "run_async"]:
        method = getattr(LlmJudge, method_name, None)
        if method:
            parts.append(f"### `LlmJudge.{method_name}()`")
            parts.append("")
            method_parsed = _parse_docstring(method.__doc__)
            if method_parsed["summary"]:
                parts.append(method_parsed["summary"])
                parts.append("")
            parts.append(_get_signature_block(method, method_name))
            table = _param_table(method, method_parsed)
            if table:
                parts.append("")
                parts.append(table)
            if method_parsed["returns"]:
                parts.append("")
                parts.append(f"**Returns:** {method_parsed['returns']}")
            parts.append("")

    # AgentJudge
    parts.append("---")
    parts.append("")
    parts.append("## AgentJudge")
    parts.append("")
    parsed = _parse_docstring(AgentJudge.__doc__)
    if parsed["summary"]:
        parts.append(parsed["summary"])
        parts.append("")
    parts.append("### Constructor")
    parts.append("")
    init_parsed = _parse_docstring(AgentJudge.__init__.__doc__)
    parts.append(_get_signature_block(AgentJudge.__init__, "__init__"))
    table = _param_table(AgentJudge.__init__, init_parsed)
    if table:
        parts.append("")
        parts.append(table)
    parts.append("")

    for method_name in ["run", "run_async"]:
        method = getattr(AgentJudge, method_name, None)
        if method:
            parts.append(f"### `AgentJudge.{method_name}()`")
            parts.append("")
            method_parsed = _parse_docstring(method.__doc__)
            if method_parsed["summary"]:
                parts.append(method_parsed["summary"])
                parts.append("")
            parts.append(_get_signature_block(method, method_name))
            table = _param_table(method, method_parsed)
            if table:
                parts.append("")
                parts.append(table)
            if method_parsed["returns"]:
                parts.append("")
                parts.append(f"**Returns:** {method_parsed['returns']}")
            parts.append("")

    return "\n".join(parts)


def generate_tools_page() -> str:
    import tinyagent.tools as tools_module

    parts = [
        "---",
        "title: Tools",
        "description: Built-in tools provided by tinyagent",
        "---",
        "",
        "Built-in callable tools that can be passed directly to `AgentConfig.tools`.",
        "",
    ]

    public_tools = [
        name
        for name in sorted(dir(tools_module))
        if not name.startswith("_")
        and callable(getattr(tools_module, name))
        and not inspect.isclass(getattr(tools_module, name))
        and name in getattr(tools_module, "__all__", [])
    ]

    for tool_name in public_tools:
        tool_func = getattr(tools_module, tool_name)
        parts.append(
            _generate_function_doc(tool_func, "##", "tinyagent.tools", tool_name)
        )
        parts.append("")

    return "\n".join(parts)


def generate_serving_page() -> str:
    parts = [
        "---",
        "title: Serving",
        "description: ServerHandle reference",
        "---",
        "",
    ]

    try:
        from tinyagent.serving import ServerHandle

        parts.append("## ServerHandle")
        parts.append("")
        parsed = _parse_docstring(ServerHandle.__doc__)
        if parsed["summary"]:
            parts.append(parsed["summary"])
            parts.append("")
        parts.append(
            "Lifecycle management for async servers returned by `TinyAgent.serve_async()`."
        )
        parts.append("")
        parts.append("### Fields")
        parts.append("")
        parts.append("| Field | Type | Description |")
        parts.append("|-------|------|-------------|")
        parts.append("| `task` | `asyncio.Task` | The server task |")
        parts.append("| `server` | `UvicornServer` | The uvicorn server instance |")
        parts.append("")

        for method_name in ["shutdown", "is_running"]:
            method = getattr(ServerHandle, method_name, None)
            if method:
                parts.append(f"### `ServerHandle.{method_name}()`")
                parts.append("")
                method_parsed = _parse_docstring(method.__doc__)
                if method_parsed["summary"]:
                    parts.append(method_parsed["summary"])
                    parts.append("")
                parts.append(_get_signature_block(method, method_name))
                parts.append("")

        parts.append("### Properties")
        parts.append("")
        parts.append(
            "- `port` - `int`: The actual server port (useful when port=0 for OS-assigned ports)."
        )
        parts.append("")
    except ImportError:
        parts.append("## ServerHandle")
        parts.append("")
        parts.append(
            "Lifecycle management for async servers. Install the full package for complete docs."
        )
        parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


PAGES: dict[str, Any] = {
    "agent.md": generate_agent_page,
    "config.md": generate_config_page,
    "callbacks.md": generate_callbacks_page,
    "tracing.md": generate_tracing_page,
    "evaluation.md": generate_evaluation_page,
    "tools.md": generate_tools_page,
    "serving.md": generate_serving_page,
}


def main() -> int:
    DOCS_API_DIR.mkdir(parents=True, exist_ok=True)
    for filename, generator in PAGES.items():
        dest = DOCS_API_DIR / filename
        content = generator()
        dest.write_text(content, encoding="utf-8")
        print(f"Generated {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
