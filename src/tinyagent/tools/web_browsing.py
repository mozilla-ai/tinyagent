import json
import os
import re
from typing import Any

import requests
from requests.exceptions import RequestException

_YOUCOM_FREE_MCP_URL = "https://api.you.com/mcp?profile=free"
_YOUCOM_MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


def _truncate_content(content: str, max_length: int) -> str:
    if len(content) <= max_length:
        return content
    return (
        content[: max_length // 2]
        + f"\n..._This content has been truncated to stay below {max_length} characters_...\n"
        + content[-max_length // 2 :]
    )


def search_web(query: str) -> str:
    """Perform a duckduckgo web search based on your query (think a Google search) then returns the top search results.

    Args:
        query (str): The search query to perform.

    Returns:
        The top search results.

    """
    try:
        from duckduckgo_search import DDGS  # type: ignore[import-not-found]
    except ImportError as e:
        msg = "You need to `pip install 'duckduckgo_search'` to use this tool"
        raise ImportError(msg) from e

    ddgs = DDGS()
    results = ddgs.text(query, max_results=10)
    return "\n".join(
        f"[{result['title']}]({result['href']})\n{result['body']}" for result in results
    )


def visit_webpage(url: str, timeout: int = 30, max_length: int = 10000) -> str:
    """Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages.

    Args:
        url: The url of the webpage to visit.
        timeout: The timeout in seconds for the request.
        max_length: The maximum number of characters of text that can be returned (default=10000).
                    If max_length==-1, text is not truncated and the full webpage is returned.

    """
    try:
        from markdownify import markdownify  # type: ignore[import-not-found]
    except ImportError as e:
        msg = "You need to `pip install 'markdownify'` to use this tool"
        raise ImportError(msg) from e

    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        markdown_content = markdownify(response.text).strip()

        markdown_content = re.sub(r"\n{2,}", "\n", markdown_content)

        if max_length == -1:
            return str(markdown_content)
        return _truncate_content(markdown_content, max_length)
    except RequestException as e:
        return f"Error fetching the webpage: {e!s}"
    except Exception as e:
        return f"An unexpected error occurred: {e!s}"


def search_tavily(query: str, include_images: bool = False) -> str:
    """Perform a Tavily web search based on your query and return the top search results.

    See https://blog.tavily.com/getting-started-with-the-tavily-search-api for more information.

    Args:
        query (str): The search query to perform.
        include_images (bool): Whether to include images in the results.

    Returns:
        The top search results as a formatted string.

    """
    try:
        from tavily.tavily import TavilyClient
    except ImportError as e:
        msg = "You need to `pip install 'tavily-python'` to use this tool"
        raise ImportError(msg) from e

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "TAVILY_API_KEY environment variable not set."
    try:
        client = TavilyClient(api_key)
        response = client.search(query, include_images=include_images)
        results = response.get("results", [])
        output = []
        for result in results:
            output.append(
                f"[{result.get('title', 'No Title')}]({result.get('url', '#')})\n{result.get('content', '')}"
            )
        if include_images and "images" in response:
            output.append("\nImages:")
            for image in response["images"]:
                output.append(image)
        return "\n\n".join(output) if output else "No results found."
    except Exception as e:
        return f"Error performing Tavily search: {e!s}"


def _parse_mcp_response(response: requests.Response) -> dict[str, Any]:
    """Parse a single JSON-RPC message from an MCP response body (JSON or SSE)."""
    content_type = response.headers.get("Content-Type", "")
    if "text/event-stream" in content_type:
        for line in reversed(response.text.splitlines()):
            if line.startswith("data:"):
                message: dict[str, Any] = json.loads(line.removeprefix("data:").strip())
                return message
        msg = "No JSON-RPC message found in the MCP response"
        raise ValueError(msg)
    return dict(response.json())


def _youcom_mcp_call(tool: str, arguments: dict[str, Any], timeout: int = 30) -> str:
    """Call a tool on You.com's keyless MCP free profile and return its text content."""
    from tinyagent import __version__

    headers = dict(_YOUCOM_MCP_HEADERS)
    initialize = requests.post(
        _YOUCOM_FREE_MCP_URL,
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "tinyagent", "version": __version__},
            },
        },
        timeout=timeout,
    )
    initialize.raise_for_status()
    session_id = initialize.headers.get("Mcp-Session-Id")
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    notification = requests.post(
        _YOUCOM_FREE_MCP_URL,
        headers=headers,
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        timeout=timeout,
    )
    notification.raise_for_status()
    response = requests.post(
        _YOUCOM_FREE_MCP_URL,
        headers=headers,
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    message = _parse_mcp_response(response)
    if "error" in message:
        msg = str(message["error"])
        raise ValueError(msg)
    result: dict[str, Any] = message.get("result", {})
    if result.get("isError"):
        content = result.get("content") or [{"text": "unknown error"}]
        msg = str(content[0].get("text", "unknown error"))
        raise ValueError(msg)
    content = result.get("content") or []
    if not content:
        msg = "No content returned by the You.com MCP tool"
        raise ValueError(msg)
    return str(content[0].get("text", ""))


def search_youcom(query: str, max_results: int = 10, timeout: int = 30) -> str:
    """Perform a You.com web search based on your query and return the top search results.

    Uses You.com's keyless MCP free profile (https://api.you.com/mcp?profile=free), so no API
    key or signup is required. See https://github.com/youdotcom-oss/agent-skills for more
    You.com integrations.

    Args:
        query (str): The search query to perform.
        max_results (int): The maximum number of results to return (default=10).
        timeout (int): The timeout in seconds for each HTTP request (default=30).

    Returns:
        The top search results as a formatted string.

    """
    try:
        text = _youcom_mcp_call(
            "you-search",
            {"query": query, "count": max(1, max_results)},
            timeout=timeout,
        )
        data: dict[str, Any] = json.loads(text)
        results = data.get("results", {}).get("web", [])
        output = []
        for result in results[: max(1, max_results)]:
            snippet = result.get("description")
            if not snippet:
                highlights = (result.get("contents") or {}).get("highlights") or []
                snippet = highlights[0] if highlights else ""
            output.append(
                f"[{result.get('title', 'No Title')}]({result.get('url', '#')})\n{snippet}"
            )
        return "\n\n".join(output) if output else "No results found."
    except Exception as e:
        return f"Error performing You.com search: {e!s}"
