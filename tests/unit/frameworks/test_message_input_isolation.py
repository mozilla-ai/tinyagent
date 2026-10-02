from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.completion import ChatCompletion

from tinyagent import AgentConfig, TinyAgent
from tinyagent.agent import AgentRunError
from tinyagent.testing.helpers import DEFAULT_SMALL_MODEL_ID, LLM_IMPORT_PATH


def _response(
    *, tool_call: bool = False, tool_name: str = "final_answer"
) -> ChatCompletion:
    message: dict[str, Any] = {"role": "assistant", "content": "done"}
    if tool_call:
        message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call-final",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": '{"answer": "done"}',
                    },
                }
            ],
        }
    return ChatCompletion.model_validate(
        {
            "id": "completion-test",
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        }
    )


@pytest.mark.parametrize("tool_call", [False, True])
async def test_run_preserves_caller_messages(tool_call: bool) -> None:
    agent = await TinyAgent.create_async(AgentConfig(model_id=DEFAULT_SMALL_MODEL_ID))
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "Custom instructions"},
        {"role": "user", "content": [{"type": "text", "text": "hello"}]},
    ]
    original = deepcopy(messages)
    with patch(
        LLM_IMPORT_PATH, new=AsyncMock(return_value=_response(tool_call=tool_call))
    ):
        result = await agent.run_async(messages)

    assert result.final_output == "done"
    assert messages == original


async def test_reusing_messages_starts_each_run_from_original_input() -> None:
    agent = await TinyAgent.create_async(AgentConfig(model_id=DEFAULT_SMALL_MODEL_ID))
    messages = [{"role": "user", "content": "hello"}]
    original = deepcopy(messages)
    requests: list[list[dict[str, Any]]] = []

    async def complete(**kwargs: Any) -> ChatCompletion:
        requests.append(deepcopy(kwargs["messages"]))
        return _response()

    with patch(LLM_IMPORT_PATH, side_effect=complete):
        await agent.run_async(messages)
        await agent.run_async(messages)

    assert requests == [original, original]
    assert messages == original


async def test_failed_run_preserves_caller_messages() -> None:
    agent = await TinyAgent.create_async(AgentConfig(model_id=DEFAULT_SMALL_MODEL_ID))
    messages = [{"role": "user", "content": "hello"}]
    original = deepcopy(messages)
    response = _response(tool_call=True, tool_name="missing_tool")
    with patch(LLM_IMPORT_PATH, side_effect=[response, RuntimeError("model failed")]):
        with pytest.raises(AgentRunError, match="model failed"):
            await agent.run_async(messages)

    assert messages == original
