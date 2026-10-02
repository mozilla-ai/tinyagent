from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from any_llm.types.completion import ChatCompletion

from tinyagent import AgentConfig, TinyAgent
from tinyagent.testing.helpers import DEFAULT_SMALL_MODEL_ID, LLM_IMPORT_PATH


def _response(arguments: str, tool_name: str = "echo") -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "completion-test",
            "created": 0,
            "model": "test-model",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call-test",
                                "type": "function",
                                "function": {"name": tool_name, "arguments": arguments},
                            }
                        ],
                    },
                }
            ],
        }
    )


@pytest.mark.parametrize(
    "arguments", ['{"text":', "null", "[]", '"hello"', "42", "true"]
)
@pytest.mark.parametrize("tool_name", ["echo", "final_answer"])
async def test_invalid_arguments_are_returned_to_model(
    arguments: str, tool_name: str
) -> None:
    calls: list[str] = []

    def echo(text: str) -> str:
        """Echo text for the argument validation test."""
        calls.append(text)
        return text

    agent = await TinyAgent.create_async(
        AgentConfig(model_id=DEFAULT_SMALL_MODEL_ID, tools=[echo])
    )
    bad_response = _response(arguments, tool_name)
    corrected_response = _response('{"text": "recovered"}')
    final_response = _response('{"answer": "done"}', "final_answer")
    responses = iter([bad_response, corrected_response, final_response])
    requests: list[list[dict[str, Any]]] = []

    async def complete(**kwargs: Any) -> ChatCompletion:
        requests.append(deepcopy(kwargs["messages"]))
        return next(responses)

    with patch(LLM_IMPORT_PATH, side_effect=complete):
        result = await agent.run_async("echo recovered")

    assert result.final_output == "done"
    assert calls == ["recovered"]
    assert len(requests) == 3
    feedback = requests[1][-1]
    assert feedback["role"] == "tool"
    assert feedback["tool_call_id"] == "call-test"
    assert feedback["name"] == tool_name
    assert feedback["content"].startswith("Error calling tool:")
    assert "JSON object" in feedback["content"]
    assert requests[2][-1]["content"] == "recovered"


@pytest.mark.parametrize("tool_name", ["echo", "final_answer"])
async def test_invalid_call_does_not_skip_valid_calls_in_same_response(
    tool_name: str,
) -> None:
    calls: list[str] = []

    def echo(text: str) -> str:
        """Echo text for the mixed tool batch test."""
        calls.append(text)
        return text

    agent = await TinyAgent.create_async(
        AgentConfig(model_id=DEFAULT_SMALL_MODEL_ID, tools=[echo])
    )
    response_data = _response("null", tool_name).model_dump()
    valid_call = _response('{"text": "same batch"}').model_dump()["choices"][0][
        "message"
    ]["tool_calls"][0]
    valid_call["id"] = "call-valid"
    response_data["choices"][0]["message"]["tool_calls"].append(valid_call)
    responses = iter(
        [
            ChatCompletion.model_validate(response_data),
            _response('{"answer": "done"}', "final_answer"),
        ]
    )
    requests: list[list[dict[str, Any]]] = []

    async def complete(**kwargs: Any) -> ChatCompletion:
        requests.append(deepcopy(kwargs["messages"]))
        return next(responses)

    with patch(LLM_IMPORT_PATH, side_effect=complete):
        result = await agent.run_async("echo same batch")

    assert result.final_output == "done"
    assert calls == ["same batch"]
    assert len(requests) == 2
    assert requests[1][-2]["content"].startswith("Error calling tool:")
    assert requests[1][-2]["tool_call_id"] == "call-test"
    assert requests[1][-1]["content"] == "same batch"
    assert requests[1][-1]["tool_call_id"] == "call-valid"


@pytest.mark.parametrize("arguments", ["", "{}"])
async def test_empty_arguments_still_call_zero_argument_tool(arguments: str) -> None:
    calls: list[bool] = []

    def no_arguments() -> str:
        """Record a zero-argument tool invocation."""
        calls.append(True)
        return "called"

    agent = await TinyAgent.create_async(
        AgentConfig(model_id=DEFAULT_SMALL_MODEL_ID, tools=[no_arguments])
    )
    with patch(
        LLM_IMPORT_PATH,
        new=AsyncMock(
            side_effect=[
                _response(arguments, "no_arguments"),
                _response('{"answer": "done"}', "final_answer"),
            ]
        ),
    ):
        result = await agent.run_async("call no_arguments")

    assert result.final_output == "done"
    assert calls == [True]
