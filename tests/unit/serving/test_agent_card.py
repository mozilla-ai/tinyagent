from unittest.mock import MagicMock

import pytest

from tinyagent import AgentConfig
from tinyagent.config import MCPSse
from tinyagent.tools import _wrap_tools, search_web

pytest.importorskip("a2a")


from a2a.types import AgentSkill

from tinyagent.serving import A2AServingConfig
from tinyagent.serving.a2a.agent_card import _get_agent_card


def test_get_agent_card() -> None:
    agent = MagicMock()
    agent.config = AgentConfig(model_id="mistral:foo", description="test agent")
    agent._tools = [search_web]
    agent_card = _get_agent_card(agent, A2AServingConfig())
    assert agent_card.name == "tinyagent"
    assert agent_card.description == "test agent"
    assert len(agent_card.skills) == 1
    assert agent_card.skills[0].id == "tinyagent-search_web"
    assert agent_card.skills[0].name == "search_web"
    assert "Perform a duckduckgo web search" in agent_card.skills[0].description
    assert not agent_card.capabilities.streaming
    assert agent_card.capabilities.push_notifications
    assert not agent_card.capabilities.state_transition_history
    assert agent_card.url == "http://localhost:5000/"


@pytest.mark.asyncio
async def test_get_agent_card_with_mcp(echo_sse_server) -> None:  # type: ignore[no-untyped-def]
    agent = MagicMock()
    agent.config = AgentConfig(model_id="mistral:foo", description="test agent")

    mcp_config = MCPSse(url=echo_sse_server["url"])
    wrapped_tools, _ = await _wrap_tools([mcp_config])
    agent._tools = wrapped_tools

    agent_card = _get_agent_card(agent, A2AServingConfig())
    assert agent_card.name == "tinyagent"
    assert agent_card.description == "test agent"
    assert len(agent_card.skills) == 3
    assert agent_card.skills[0].id == "tinyagent-write_file"
    assert agent_card.skills[0].name == "write_file"
    assert "Say hi back with the input text" in agent_card.skills[0].description


def test_get_agent_card_with_explicit_skills() -> None:
    """When skills are explicitly provided in A2AServingConfig, they are used instead of inferring from tools."""
    agent = MagicMock()
    agent.config = AgentConfig(model_id="mistral:foo", description="test agent")
    agent._tools = [search_web]

    explicit_skills = [
        AgentSkill(
            id="custom-skill-1",
            name="custom_function_1",
            description="This is a custom skill that does something amazing",
            tags=["custom", "test"],
        ),
        AgentSkill(
            id="custom-skill-2",
            name="custom_function_2",
            description="Another custom skill for testing purposes",
            tags=["custom", "demo"],
        ),
    ]

    serving_config = A2AServingConfig(skills=explicit_skills)
    agent_card = _get_agent_card(agent, serving_config)

    assert agent_card.name == "tinyagent"
    assert agent_card.description == "test agent"

    assert len(agent_card.skills) == 2
    assert agent_card.skills[0].id == "custom-skill-1"
    assert agent_card.skills[1].id == "custom-skill-2"

    skill_names = [skill.name for skill in agent_card.skills]
    assert "search_web" not in skill_names
