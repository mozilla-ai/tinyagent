"""TinyAgent driving an MCP server over the streamable-HTTP transport."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from tinyagent import AgentConfig, TinyAgent
from tinyagent.config import MCPStreamableHttp
from tinyagent.testing.helpers import (
    DEFAULT_SMALL_MODEL_ID,
    get_default_agent_model_args,
    group_spans,
)


pytestmark = pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"), reason="MISTRAL_API_KEY not set"
)


class Step(BaseModel):
    number: int
    description: str


class Steps(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: list[Step]


def test_load_and_run_agent_streamable_http(
    date_streamable_http_server: dict[str, Any],
    tmp_path: Path,
) -> None:
    tmp_file = "tmp.txt"

    def write_file(text: str) -> None:
        """Write the text to a file.

        Args:
            text: The text to write.

        Returns:
            None.

        """
        (tmp_path / tmp_file).write_text(text)

    tools = [
        write_file,
        MCPStreamableHttp(
            url=date_streamable_http_server["url"],
            client_session_timeout_seconds=30,
        ),
    ]
    agent_config = AgentConfig(
        model_id=DEFAULT_SMALL_MODEL_ID,
        tools=tools,  # type: ignore[arg-type]
        instructions="Use the available tools to answer.",
        model_args=get_default_agent_model_args(model_id=DEFAULT_SMALL_MODEL_ID),
        output_type=Steps,
    )
    agent = TinyAgent.create(agent_config)

    agent_trace = agent.run(
        "First, find what year it is in the America/New_York timezone. "
        "Then, write the value (single number) to a file. "
        "Finally, return a list of the steps you have taken.",
    )

    assert isinstance(agent_trace.final_output, Steps)

    assert (tmp_path / tmp_file).read_text() == str(datetime.now().year)
    _, _, tool_executions = group_spans(agent_trace.spans)

    assert len(tool_executions) >= 1
    tool_args_raw = tool_executions[0].attributes.get("gen_ai.tool.args")
    assert tool_args_raw is not None
    args = json.loads(tool_args_raw)
    assert "timezone" in args
