import json
from typing import Any
from unittest.mock import MagicMock, patch


def test_search_tavily_no_api_key(monkeypatch: Any) -> None:
    with patch("tavily.tavily.TavilyClient", MagicMock()):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        from tinyagent.tools import search_tavily

        result = search_tavily("test")
        assert "environment variable not set" in result


def test_search_tavily_success(monkeypatch: Any) -> None:
    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def search(self, query: str, include_images: bool = False) -> Any:
            return {
                "results": [
                    {
                        "title": "Test Title",
                        "url": "http://test.com",
                        "content": "Test content!",
                    }
                ]
            }

    with patch("tavily.tavily.TavilyClient", FakeClient):
        monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
        from tinyagent.tools import search_tavily

        result = search_tavily("test")
        assert "Test Title" in result
        assert "Test content!" in result


def test_search_tavily_with_images(monkeypatch: Any) -> None:
    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def search(self, query: str, include_images: bool = False) -> Any:
            return {
                "results": [
                    {
                        "title": "Test Title",
                        "url": "http://test.com",
                        "content": "Test content!",
                    }
                ],
                "images": ["http://image.com/cat.jpg"],
            }

    with patch("tavily.tavily.TavilyClient", FakeClient):
        monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
        from tinyagent.tools import search_tavily

        result = search_tavily("test", include_images=True)
        assert "Test Title" in result
        assert "Images:" in result
        assert "cat.jpg" in result


def test_search_tavily_exception(monkeypatch: Any) -> None:
    class FakeClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key

        def search(self, query: str, include_images: bool = False) -> Any:
            msg = "Oops!"
            raise RuntimeError(msg)

    with patch("tavily.tavily.TavilyClient", FakeClient):
        monkeypatch.setenv("TAVILY_API_KEY", "fake-key")
        from tinyagent.tools import search_tavily

        result = search_tavily("test")
        assert "Error performing Tavily search" in result


class _FakeMcpResponse:
    """Minimal stand-in for a requests.Response carrying a single JSON-RPC message."""

    def __init__(
        self, message: dict[str, Any], content_type: str = "text/event-stream"
    ) -> None:
        self.message = message
        self.headers = {"Content-Type": content_type}
        if "text/event-stream" in content_type:
            self.text = f"data: {json.dumps(message)}\n\n"
        else:
            self.text = json.dumps(message)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.message


def _fake_mcp_tool_response(
    payload: dict[str, Any], content_type: str = "text/event-stream"
) -> list[Any]:
    handshake = _FakeMcpResponse(
        {"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2025-03-26"}}
    )
    tool_call = _FakeMcpResponse(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "content": [{"type": "text", "text": json.dumps(payload)}],
            },
        },
        content_type=content_type,
    )
    return [handshake, handshake, tool_call]


def test_search_youcom_success(monkeypatch: Any) -> None:
    payload = {
        "results": {
            "web": [
                {
                    "title": "Test Title",
                    "url": "http://test.com",
                    "description": "Test content!",
                    "contents": {},
                }
            ]
        }
    }

    with (
        patch(
            "tinyagent.tools.web_browsing.requests.post",
            MagicMock(side_effect=_fake_mcp_tool_response(payload)),
        ),
    ):
        from tinyagent.tools import search_youcom

        result = search_youcom("test")
        assert "Test Title" in result
        assert "http://test.com" in result
        assert "Test content!" in result


def test_search_youcom_uses_highlights_when_no_description(monkeypatch: Any) -> None:
    payload = {
        "results": {
            "web": [
                {
                    "title": "Test Title",
                    "url": "http://test.com",
                    "description": "",
                    "contents": {"highlights": ["First highlight passage"]},
                }
            ]
        }
    }

    with (
        patch(
            "tinyagent.tools.web_browsing.requests.post",
            MagicMock(side_effect=_fake_mcp_tool_response(payload)),
        ),
    ):
        from tinyagent.tools import search_youcom

        result = search_youcom("test")
        assert "Test Title" in result
        assert "First highlight passage" in result


def test_search_youcom_no_results(monkeypatch: Any) -> None:
    payload: dict[str, Any] = {"results": {"web": []}}

    with (
        patch(
            "tinyagent.tools.web_browsing.requests.post",
            MagicMock(
                side_effect=_fake_mcp_tool_response(
                    payload, content_type="application/json"
                )
            ),
        ),
    ):
        from tinyagent.tools import search_youcom

        result = search_youcom("test")
        assert result == "No results found."


def test_search_youcom_exception(monkeypatch: Any) -> None:
    with patch(
        "tinyagent.tools.web_browsing.requests.post",
        MagicMock(side_effect=RuntimeError("Oops!")),
    ):
        from tinyagent.tools import search_youcom

        result = search_youcom("test")
        assert "Error performing You.com search" in result
