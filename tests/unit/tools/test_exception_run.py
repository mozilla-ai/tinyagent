from collections.abc import Iterator
from unittest.mock import patch

from any_llm.types.completion import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    Choice,
    CompletionUsage,
    Function as AnyllmFunction,
)

from tinyagent import AgentConfig, TinyAgent
from tinyagent.testing.helpers import LLM_IMPORT_PATH
from tinyagent.tracing.otel_types import StatusCode


def test_tool_error_llm_mocked() -> None:
    """An exception raised inside a tool is caught and surfaced as an ERROR span.

    The agent should be allowed to try to recover from the failure rather than crash.
    """
    exc_reason = "It's a trap!"

    def search_web(query: str) -> str:
        """Perform a duckduckgo web search based on your query.

        Args:
            query: The search query to perform.

        Returns:
            The top search results.

        """
        msg = exc_reason
        raise ValueError(msg)

    agent_config = AgentConfig(
        model_id="mistral:mistral-small-latest",
        instructions="You must use the available tools to answer questions.",
        tools=[search_web],
        model_args={"temperature": 0.0},
    )

    agent = TinyAgent.create(agent_config)

    give_up = "I am unable to perform web searches; here is some general information."

    fake_tool_fail_response = ChatCompletion(
        id="chatcmpl-tool-fail",
        created=1747157127,
        model="mistral-small-latest",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(
                    content="",
                    role="assistant",
                    tool_calls=[
                        ChatCompletionMessageFunctionToolCall(
                            id="call_12345xyz",
                            type="function",
                            function=AnyllmFunction(
                                name="search_web",
                                arguments='{"query": "which agent framework is the best"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    fake_give_up_response = ChatCompletion(
        id="chatcmpl-give-up",
        created=1747157127,
        model="mistral-small-latest",
        object="chat.completion",
        choices=[
            Choice(
                finish_reason="tool_calls",
                index=0,
                message=ChatCompletionMessage(
                    content="",
                    role="assistant",
                    tool_calls=[
                        ChatCompletionMessageFunctionToolCall(
                            id="call_final",
                            type="function",
                            function=AnyllmFunction(
                                name="final_answer",
                                arguments='{"answer": "' + give_up + '"}',
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=CompletionUsage(prompt_tokens=20, completion_tokens=50, total_tokens=70),
    )

    def mock_generator() -> Iterator[ChatCompletion]:
        yield fake_tool_fail_response
        while True:
            yield fake_give_up_response

    with patch(LLM_IMPORT_PATH) as llm_mock:
        llm_mock.side_effect = mock_generator()

        agent_trace = agent.run(
            "Check in the web which agent framework is the best.",
        )

        assert any(
            span.is_tool_execution()
            and span.status.status_code == StatusCode.ERROR
            and exc_reason in getattr(span.status, "description", "")
            for span in agent_trace.spans
        )
