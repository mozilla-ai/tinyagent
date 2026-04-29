from typing import Any

import pytest
from pydantic import BaseModel

pytest.importorskip("a2a")

from a2a.types import TaskState

from tinyagent.agent import TinyAgent
from tinyagent.config import AgentConfig
from tinyagent.serving.a2a.envelope import (
    A2AEnvelope,
    _DefaultBody,
    _is_a2a_envelope,
    prepare_agent_for_a2a_async,
)


class CustomOutputType(BaseModel):
    custom_field: str
    result: str


class MockAgent(TinyAgent):
    """Mock agent implementation for testing the A2A envelope wiring."""

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        self._agent = None

    async def _load_agent(self) -> None:
        pass

    async def _run_async(
        self, prompt: str | list[dict[str, Any]], **kwargs: Any
    ) -> str:
        return "mock result"

    async def update_output_type_async(
        self, output_type: type[BaseModel] | None
    ) -> None:
        self.config.output_type = output_type


@pytest.mark.asyncio
async def test_envelope_created_without_output_type() -> None:
    """Test that the envelope is correctly created when the agent is configured without an output_type."""
    config = AgentConfig(model_id="mistral:test-model", description="test agent")
    assert config.output_type is None

    agent = MockAgent(config)

    prepared_agent = await prepare_agent_for_a2a_async(agent)

    assert prepared_agent.config.output_type is not None
    assert _is_a2a_envelope(prepared_agent.config.output_type)

    envelope_instance = prepared_agent.config.output_type(
        task_status=TaskState.completed, data=_DefaultBody(result="test result")
    )

    assert isinstance(envelope_instance, A2AEnvelope)
    assert envelope_instance.task_status == TaskState.completed
    assert isinstance(envelope_instance.data, _DefaultBody)
    assert envelope_instance.data.result == "test result"


@pytest.mark.asyncio
async def test_envelope_created_with_output_type() -> None:
    """Test that the envelope is correctly created when an agent is configured with an output_type."""
    config = AgentConfig(
        model_id="mistral:test-model", description="test agent", output_type=CustomOutputType
    )

    agent = MockAgent(config)

    prepared_agent = await prepare_agent_for_a2a_async(agent)

    assert prepared_agent.config.output_type is not None
    assert _is_a2a_envelope(prepared_agent.config.output_type)

    envelope_instance = prepared_agent.config.output_type(
        task_status=TaskState.completed,
        data=CustomOutputType(custom_field="test", result="custom result"),
    )

    assert isinstance(envelope_instance, A2AEnvelope)
    assert envelope_instance.task_status == TaskState.completed
    assert isinstance(envelope_instance.data, CustomOutputType)
    assert envelope_instance.data.custom_field == "test"
    assert envelope_instance.data.result == "custom result"
