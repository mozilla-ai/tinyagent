from unittest.mock import MagicMock

import pytest

from tinyagent.callbacks.wrapper import _TinyAgentWrapper


@pytest.mark.asyncio
async def test_unwrap_before_wrap() -> None:
    """`unwrap` should be safe to call before `wrap`."""
    wrapper = _TinyAgentWrapper()
    await wrapper.unwrap(MagicMock())


@pytest.mark.asyncio
async def test_wrap_then_unwrap_restores_original_call_model() -> None:
    """After `wrap`, `agent.call_model` is patched; `unwrap` restores the original."""
    wrapper = _TinyAgentWrapper()

    original_call_model = MagicMock(name="original_call_model")
    agent = MagicMock()
    agent.call_model = original_call_model
    agent.clients = {}
    agent.config.callbacks = []

    await wrapper.wrap(agent)
    assert agent.call_model is not original_call_model

    await wrapper.unwrap(agent)
    assert agent.call_model is original_call_model
