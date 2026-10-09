"""Per-channel overrides for the progress / tool-hint progress stream."""

from __future__ import annotations

import asyncio
import contextlib
from unittest.mock import AsyncMock

import pytest

from openharness.channels.bus.events import OutboundMessage
from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.manager import ChannelManager
from openharness.config.schema import Config, QQConfig


class _FakeChannel:
    def __init__(self, config):
        self.config = config
        self.send = AsyncMock()


def _message(*, progress: bool, hint: bool = False) -> OutboundMessage:
    metadata: dict[str, object] = {}
    if progress:
        metadata["_progress"] = True
    if hint:
        metadata["_tool_hint"] = True
    return OutboundMessage(channel="qq", chat_id="g1", content="x", metadata=metadata)


async def _dispatch(config: Config, channel_config, messages) -> list[OutboundMessage]:
    manager = ChannelManager(config, MessageBus())
    channel = _FakeChannel(channel_config)
    manager.channels["qq"] = channel  # type: ignore[assignment]

    task = asyncio.create_task(manager._dispatch_outbound())
    try:
        for message in messages:
            await manager.bus.publish_outbound(message)
        deadline = asyncio.get_running_loop().time() + 2.0
        while manager.bus.outbound_size and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        # One more loop tick so a filtered message cannot land after we look.
        await asyncio.sleep(0.05)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    return [call.args[0] for call in channel.send.await_args_list]


@pytest.mark.asyncio
async def test_channel_override_silences_progress_without_touching_the_global():
    config = Config()  # container-wide send_progress / send_tool_hints are True
    final = _message(progress=False)

    sent = await _dispatch(
        config,
        QQConfig(send_progress=False, send_tool_hints=False),
        [_message(progress=True), _message(progress=True, hint=True), final],
    )

    assert sent == [final]
    assert config.channels.send_progress is True
    assert config.channels.send_tool_hints is True


@pytest.mark.asyncio
async def test_unset_override_inherits_the_global_setting():
    config = Config()
    config.channels.send_progress = False

    sent = await _dispatch(config, QQConfig(), [_message(progress=True), _message(progress=False)])

    # Inherited False drops progress, but the real reply still goes out.
    assert [message.content for message in sent] == ["x"]
    assert sent[0].metadata == {}


@pytest.mark.asyncio
async def test_explicit_override_beats_the_global_in_both_directions():
    config = Config()
    config.channels.send_tool_hints = False

    sent = await _dispatch(
        config,
        QQConfig(send_progress=False, send_tool_hints=True),
        [_message(progress=True, hint=True), _message(progress=True), _message(progress=False)],
    )

    # The per-channel hint override wins over the global False -- the global is a
    # fallback for None, not a ceiling -- while the per-channel progress override
    # still drops the plain progress beat after it.
    assert [message.metadata.get("_tool_hint") for message in sent] == [True, None]


def test_qq_config_accepts_the_override_keys():
    config = QQConfig.model_validate(
        {"app_id": "1", "app_secret": "s", "allow_from": ["*"], "send_progress": False}
    )
    assert config.send_progress is False
    assert config.send_tool_hints is None


def test_unrelated_channel_configs_default_to_inheriting():
    from openharness.config.schema import FeishuConfig, TelegramConfig

    for channel_config in (FeishuConfig(), TelegramConfig()):
        assert channel_config.send_progress is None
        assert channel_config.send_tool_hints is None
