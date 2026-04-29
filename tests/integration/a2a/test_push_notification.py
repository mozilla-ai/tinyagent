"""A2A push-notification (webhook) flow."""

import asyncio
import json
import os
from typing import Any
from uuid import uuid4

import pytest

# `TinyAgent.__init__` constructs an LLM client which validates the API key.
# These tests mock the agent loop and never actually call the LLM, so a dummy
# value is enough to get past the validator.
os.environ.setdefault("MISTRAL_API_KEY", "dummy-mistral-key-for-tests")

pytest.importorskip("a2a")

import uvicorn
from a2a.types import (
    Message,
    MessageSendConfiguration,
    MessageSendParams,
    Part,
    PushNotificationConfig,
    Role,
    SendMessageRequest,
    Task,
    TaskState,
    TextPart,
)
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from tinyagent import AgentConfig, TinyAgent
from tinyagent.serving import A2AServingConfig
from tinyagent.serving.a2a.envelope import A2AEnvelope
from tinyagent.testing.helpers import DEFAULT_SMALL_MODEL_ID, wait_for_server_async
from tinyagent.tracing.agent_trace import AgentSpan, AgentTrace
from tinyagent.tracing.attributes import GenAI
from tinyagent.tracing.otel_types import Resource, SpanContext, SpanKind, Status

from .conftest import DEFAULT_LONG_TIMEOUT, a2a_client_from_agent

FIRST_TURN_PROMPT = "What's the capital of Pennsylvania?"
FIRST_TURN_RESPONSE = "The capital of Pennsylvania is Harrisburg."


class StringInfo(BaseModel):
    value: str


class MockConversationAgent(TinyAgent):
    """Mock TinyAgent that returns a canned A2A envelope without calling an LLM."""

    def __init__(self, config: AgentConfig) -> None:
        super().__init__(config)
        self.output_type = A2AEnvelope[StringInfo]

    async def _load_agent(self) -> None:
        await super()._load_agent()

    async def run_async(  # type: ignore[override]
        self, prompt: str | list[dict[str, Any]], instrument: bool = True, **kwargs: Any
    ) -> AgentTrace:
        envelope = self.output_type(
            task_status=TaskState.input_required,
            data=StringInfo(value=FIRST_TURN_RESPONSE),
        )
        return self._create_mock_trace(envelope, FIRST_TURN_RESPONSE, FIRST_TURN_PROMPT)

    def _create_mock_trace(
        self, envelope: A2AEnvelope[StringInfo], agent_response: str, prompt: str
    ) -> AgentTrace:
        spans = [
            AgentSpan(
                name="call_llm gpt-4o-mini",
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
async def test_push_notification_non_streaming() -> None:
    """The A2A server should POST to a configured webhook with task updates."""
    received_notifications = []

    async def webhook_handler(request: Request) -> JSONResponse:
        if request.method == "GET":
            return JSONResponse({"status": "webhook is running"}, status_code=200)

        notification_data = await request.json()
        received_notifications.append(
            {"headers": dict(request.headers), "body": notification_data}
        )

        return JSONResponse({"status": "received"}, status_code=200)

    config = AgentConfig(
        model_id=DEFAULT_SMALL_MODEL_ID,
        instructions=(
            "You are a helpful assistant that remembers our conversation. "
            "Keep your responses concise."
        ),
        description="Mock agent for push notification testing.",
    )

    agent = MockConversationAgent(config)

    serving_config = A2AServingConfig(port=0)

    webhook_app = Starlette(
        routes=[Route("/webhook", webhook_handler, methods=["GET", "POST"])]
    )

    webhook_config = uvicorn.Config(webhook_app, port=0)
    webhook_server = uvicorn.Server(webhook_config)
    webhook_task = asyncio.create_task(webhook_server.serve())

    await asyncio.sleep(0.5)
    webhook_port = webhook_server.servers[0].sockets[0].getsockname()[1]

    webhook_url = f"http://localhost:{webhook_port}/webhook"

    await wait_for_server_async(webhook_url)

    try:
        async with a2a_client_from_agent(
            agent, serving_config, http_timeout=DEFAULT_LONG_TIMEOUT
        ) as (client, _):
            first_message_id = str(uuid4())

            params = MessageSendParams(
                message=Message(
                    role=Role.user,
                    parts=[
                        Part(root=TextPart(kind="text", text=FIRST_TURN_PROMPT))
                    ],
                    message_id=first_message_id,
                ),
                configuration=MessageSendConfiguration(
                    accepted_output_modes=["text"],
                    push_notification_config=PushNotificationConfig(url=webhook_url),
                ),
            )

            request_1 = SendMessageRequest(id=str(uuid4()), params=params)
            response_1 = await client.send_message(request_1)
            if hasattr(response_1.root, "error"):
                msg = f"Error: {response_1.root.error.message}"
                raise RuntimeError(msg)
            if isinstance(response_1.root.result, Task):
                task_id = response_1.root.result.id
            else:
                task_id = response_1.root.result.message_id
            params.message.task_id = task_id

            request_1 = SendMessageRequest(id=str(uuid4()), params=params)
            response_1 = await client.send_message(request_1)
            if hasattr(response_1.root, "error"):
                msg = f"Error: {response_1.root.error.message}"
                raise RuntimeError(msg)

            response_2 = await client.send_message(request_1)
            if hasattr(response_2.root, "error"):
                msg = f"Error: {response_2.root.error.message}"
                raise RuntimeError(msg)

            await asyncio.sleep(1)

            assert len(received_notifications) == 3

    finally:
        if webhook_server:
            try:
                if hasattr(webhook_server, "shutdown"):
                    await webhook_server.shutdown()
                else:
                    webhook_server.should_exit = True
                    await asyncio.sleep(0.1)
            except Exception:
                webhook_server.should_exit = True

        if webhook_task and not webhook_task.done():
            webhook_task.cancel()
            try:
                await webhook_task
            except asyncio.CancelledError:
                pass
