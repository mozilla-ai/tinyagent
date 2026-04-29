"""Multi-turn A2A session/task management.

Mocks the agent's `run_async` so no LLM calls are made; the test verifies that
subsequent interactions correctly receive prior conversation history.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import httpx
import pytest

# `TinyAgent.__init__` constructs an LLM client which validates the API key.
# These tests mock the agent loop and never actually call the LLM, so a dummy
# value is enough to get past the validator.
os.environ.setdefault("MISTRAL_API_KEY", "dummy-mistral-key-for-tests")

pytest.importorskip("a2a")

from a2a.types import MessageSendParams, SendMessageRequest, TaskState
from pydantic import BaseModel

from tinyagent import AgentConfig, TinyAgent
from tinyagent.serving import A2AServingConfig
from tinyagent.serving.a2a.envelope import A2AEnvelope
from tinyagent.testing.helpers import (
    DEFAULT_HTTP_KWARGS,
    DEFAULT_SMALL_MODEL_ID,
    get_default_agent_model_args,
    wait_for_server_async,
)
from tinyagent.tools.a2a import a2a_tool_async
from tinyagent.tracing.agent_trace import AgentSpan, AgentTrace
from tinyagent.tracing.attributes import GenAI
from tinyagent.tracing.otel_types import Resource, SpanContext, SpanKind, Status

from .conftest import get_client_from_agent_card_url

if TYPE_CHECKING:
    from typing import Any  # noqa: F401


class UserInfo(BaseModel):
    name: str
    job: str
    age: int | None = None


FIRST_TURN_PROMPT = "My name is Alice and I work as a software engineer."
FIRST_TURN_RESPONSE = (
    "Hello, Alice! It's nice to meet you. You work as a software engineer. How old are you?"
)
SECOND_TURN_PROMPT = (
    "What's my name, what do I do for work, and what's my age? "
    "Let me know if you need more information."
)
SECOND_TURN_RESPONSE = (
    "Your name is Alice, you work as a software engineer, and your age is 30."
)
THIRD_TURN_PROMPT = "My age is 30."
THIRD_TURN_RESPONSE = "Thank you for the information."


class MockConversationAgent(TinyAgent):
    """Mock TinyAgent that simulates a multi-turn conversation without calling an LLM."""

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        self.output_type = A2AEnvelope[UserInfo]
        self.turn_count = 0

    async def _load_agent(self) -> None:
        await super()._load_agent()

    async def run_async(  # type: ignore[override]
        self, prompt: str | list[dict[str, Any]], instrument: bool = True, **kwargs: Any
    ) -> AgentTrace:
        assert isinstance(prompt, str)
        conversation_count = prompt.count("Previous conversation:")

        if self.turn_count == 0:
            assert prompt.count(FIRST_TURN_PROMPT) == 1
            assert conversation_count == 0
            self.turn_count += 1
            envelope = self.output_type(
                task_status=TaskState.completed,
                data=UserInfo(name="Alice", job="software engineer", age=None),
            )
            return self._create_mock_trace(envelope, FIRST_TURN_RESPONSE, FIRST_TURN_PROMPT)

        if self.turn_count == 1:
            assert prompt.count(FIRST_TURN_PROMPT) == 1
            assert prompt.count(SECOND_TURN_PROMPT) == 1
            assert conversation_count == 1
            self.turn_count += 1
            envelope = self.output_type(
                task_status=TaskState.input_required,
                data=UserInfo(name="Alice", job="software engineer", age=None),
            )
            return self._create_mock_trace(envelope, SECOND_TURN_RESPONSE, prompt)

        if self.turn_count == 2:
            assert prompt.count(FIRST_TURN_PROMPT) == 1
            assert prompt.count(SECOND_TURN_PROMPT) == 1
            assert prompt.count(THIRD_TURN_PROMPT) == 1
            assert conversation_count == 1
            self.turn_count += 1
            envelope = self.output_type(
                task_status=TaskState.completed,
                data=UserInfo(name="Alice", job="software engineer", age=30),
            )
            return self._create_mock_trace(envelope, THIRD_TURN_RESPONSE, prompt)

        msg = f"Unexpected turn count: {self.turn_count}"
        raise ValueError(msg)

    def _create_mock_trace(
        self, envelope: A2AEnvelope[UserInfo], agent_response: str, prompt: str
    ) -> AgentTrace:
        spans = [
            AgentSpan(
                name=f"call_llm {DEFAULT_SMALL_MODEL_ID}",
                kind=SpanKind.INTERNAL,
                status=Status(),
                context=SpanContext(span_id=123),
                attributes={
                    GenAI.OPERATION_NAME: "call_llm",
                    GenAI.REQUEST_MODEL: "mock-model",
                    GenAI.INPUT_MESSAGES: json.dumps(
                        [{"role": "user", "content": prompt}]
                    ),
                    GenAI.OUTPUT: agent_response,
                    GenAI.OUTPUT_TYPE: "json",
                },
                links=[],
                events=[],
                resource=Resource(),
            )
        ]
        return AgentTrace(spans=spans, final_output=envelope)


@pytest.mark.asyncio
async def test_a2a_tool_multiturn() -> None:
    """Agents maintain conversation context across multiple interactions via A2A."""
    config = AgentConfig(
        model_id=DEFAULT_SMALL_MODEL_ID,
        instructions=(
            "You are a helpful assistant that remembers our conversation. "
            "When asked about previous information, reference what was said earlier. "
            "Keep your responses concise."
            " If you need more information, ask the user for it."
        ),
        description="Agent with conversation memory for testing session management.",
        output_type=UserInfo,
        model_args=get_default_agent_model_args(),
    )

    agent = MockConversationAgent(config)

    serving_config = A2AServingConfig(port=0, context_timeout_minutes=2)

    server_handle = await agent.serve_async(serving_config=serving_config)
    server_url = f"http://localhost:{server_handle.port}"
    await wait_for_server_async(server_url)

    try:
        async with httpx.AsyncClient(timeout=1500) as httpx_client:
            client = await get_client_from_agent_card_url(httpx_client, server_url)

            first_message_id = str(uuid4())
            context_id = str(uuid4())

            send_message_payload_1 = {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": FIRST_TURN_PROMPT}],
                    "messageId": first_message_id,
                    "contextId": context_id,
                },
            }

            request_1 = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**send_message_payload_1),  # type: ignore[arg-type]
            )
            response_1 = await client.send_message(request_1, http_kwargs=DEFAULT_HTTP_KWARGS)

            assert response_1 is not None
            if hasattr(response_1.root, "error"):
                msg = f"Error: {response_1.root.error.message}"
                raise RuntimeError(msg)
            result = UserInfo.model_validate_json(
                response_1.root.result.status.message.parts[0].root.text
            )
            assert result.name == "Alice"
            assert result.job.lower() == "software engineer"

            send_message_payload_2 = {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": SECOND_TURN_PROMPT}],
                    "messageId": str(uuid4()),
                    "contextId": response_1.root.result.context_id,
                },
            }

            request_2 = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**send_message_payload_2),  # type: ignore[arg-type]
            )
            response_2 = await client.send_message(request_2, http_kwargs=DEFAULT_HTTP_KWARGS)

            assert response_2 is not None
            if hasattr(response_2.root, "error"):
                msg = f"Error: {response_2.root.error.message}"
                raise RuntimeError(msg)
            result = UserInfo.model_validate_json(
                response_2.root.result.status.message.parts[0].root.text
            )
            assert result.name == "Alice"
            assert result.job.lower() == "software engineer"
            assert result.age is None
            assert response_2.root.result.status.state == TaskState.input_required

            send_message_payload_3 = {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": THIRD_TURN_PROMPT}],
                    "messageId": str(uuid4()),
                    "contextId": response_2.root.result.context_id,
                    "taskId": response_2.root.result.id,
                },
            }
            request_3 = SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(**send_message_payload_3),  # type: ignore[arg-type]
            )
            response_3 = await client.send_message(request_3, http_kwargs=DEFAULT_HTTP_KWARGS)
            assert response_3 is not None
            if hasattr(response_3.root, "error"):
                msg = f"Error: {response_3.root.error.message}"
                raise RuntimeError(msg)
            result = UserInfo.model_validate_json(
                response_3.root.result.status.message.parts[0].root.text
            )
            assert response_3.root.result.status.state == TaskState.completed
            assert result.age == 30

            assert agent.turn_count == 3

    finally:
        await server_handle.shutdown()


@pytest.mark.skipif(
    not os.environ.get("MISTRAL_API_KEY"), reason="MISTRAL_API_KEY not set"
)
@pytest.mark.asyncio
async def test_a2a_tool_multiturn_async() -> None:
    """An orchestrator TinyAgent uses the conversational agent over A2A as a tool."""
    config = AgentConfig(
        model_id=DEFAULT_SMALL_MODEL_ID,
        instructions=(
            "You are a helpful assistant that remembers our conversation. "
            "When asked about previous information, reference what was said earlier. "
            "Keep your responses concise. If you need more information, ask the user for it."
        ),
        name="Structured UserInfo Agent",
        description="Agent with conversation memory for testing session management.",
        output_type=UserInfo,
        model_args=get_default_agent_model_args(),
    )

    agent = MockConversationAgent(config)

    serving_config = A2AServingConfig(port=0, context_timeout_minutes=2)

    server_handle = await agent.serve_async(serving_config=serving_config)
    server_url = f"http://localhost:{server_handle.port}"
    await wait_for_server_async(server_url)
    try:
        main_agent_cfg = AgentConfig(
            model_id=DEFAULT_SMALL_MODEL_ID,
            instructions="Use the available tools to obtain additional information to answer the query.",
            tools=[await a2a_tool_async(server_url)],
            model_args=get_default_agent_model_args(),
        )

        main_agent = await TinyAgent.create_async(main_agent_cfg)
        prompt = f"""
        Please talk to the structured UserInfo agent and interact with it. You'll contact it to ask three questions. Say the exact words from the prompt in your query to the agent.

        1. {FIRST_TURN_PROMPT}
        2. {SECOND_TURN_PROMPT}
        3. {THIRD_TURN_PROMPT}

        For question 1, when calling the tool, do not provide the context id or the task id.
        For question 2, when calling the tool, provide the context id but omit the task id.
        For question 3, when calling the tool, provide both the context id and the task id.
        """

        agent_trace = await main_agent.run_async(prompt)
        assert agent_trace.final_output is not None
        assert agent.turn_count == 3
    finally:
        await server_handle.shutdown()
