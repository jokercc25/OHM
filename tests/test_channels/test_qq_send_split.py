"""Outbound QQ text messages.

QQ rejects a message longer than 2000 characters outright (error 40054007)
instead of truncating it, so anything longer has to go out as several messages
sharing the inbound ``msg_id``. These tests pin that split, the sequence numbers
that stop the platform de-duplicating the pieces, and the reply budget's failure
mode.
"""

from __future__ import annotations

import logging

import pytest

from openharness.channels.bus.events import OutboundMessage
from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.qq import QQ_MAX_MESSAGE_LEN, QQChannel
from openharness.config.schema import QQConfig

_LOGGER = "openharness.channels.impl.qq"


class _Api:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calls: list[dict] = []
        self._fail_on_call = fail_on_call

    async def post_group_message(self, **kwargs) -> None:
        self._record(kwargs)

    async def post_c2c_message(self, **kwargs) -> None:
        self._record(kwargs)

    def _record(self, kwargs: dict) -> None:
        self.calls.append(kwargs)
        if self._fail_on_call is not None and len(self.calls) == self._fail_on_call:
            raise RuntimeError("40054007 消息长度超过限制")


class _Client:
    def __init__(self, api: _Api) -> None:
        self.api = api


def _channel(*, group: bool = True, api: _Api | None = None) -> tuple[QQChannel, _Api]:
    channel = QQChannel(QQConfig(app_id="1", app_secret="s"), MessageBus())
    api = api or _Api()
    channel._client = _Client(api)
    if group:
        channel._group_openids.add("G1")
        channel._group_msg_ids["G1"] = "MSG1"
    return channel, api


@pytest.mark.asyncio
async def test_a_short_message_goes_out_whole():
    channel, api = _channel()

    await channel.send(OutboundMessage(channel="qq", chat_id="G1", content="你好"))

    assert [call["content"] for call in api.calls] == ["你好"]


@pytest.mark.asyncio
async def test_a_long_message_is_split_under_the_platform_cap():
    channel, api = _channel()
    text = "字" * 5_000

    await channel.send(OutboundMessage(channel="qq", chat_id="G1", content=text))

    assert len(api.calls) == 3
    assert all(len(call["content"]) <= QQ_MAX_MESSAGE_LEN for call in api.calls)
    assert "".join(call["content"] for call in api.calls) == text


@pytest.mark.asyncio
async def test_split_pieces_get_increasing_sequence_numbers():
    """The sequence number is what keeps QQ from collapsing them into one."""
    channel, api = _channel()

    await channel.send(OutboundMessage(channel="qq", chat_id="G1", content="x" * 4_500))
    await channel.send(OutboundMessage(channel="qq", chat_id="G1", content="y"))

    sequences = [call["msg_seq"] for call in api.calls]
    assert sequences == [2, 3, 4, 5]


@pytest.mark.asyncio
async def test_group_replies_echo_the_msg_id_that_triggered_them():
    channel, api = _channel()

    await channel.send(OutboundMessage(channel="qq", chat_id="G1", content="你好"))

    # Passive replies must name the @-message; the channel remembers the last
    # one it saw for the group.
    assert api.calls[0]["msg_id"] == "MSG1"
    assert api.calls[0]["group_openid"] == "G1"


@pytest.mark.asyncio
async def test_a_c2c_reply_uses_the_id_off_the_outbound_message():
    channel, api = _channel(group=False)

    await channel.send(
        OutboundMessage(channel="qq", chat_id="U1", content="你好", metadata={"message_id": "M9"})
    )

    assert api.calls[0]["openid"] == "U1"
    assert api.calls[0]["msg_id"] == "M9"


@pytest.mark.asyncio
async def test_a_failed_piece_stops_the_rest(caplog):
    """The usual cause is the five-reply budget, which fails every later piece
    too -- continuing would only buy log noise."""
    channel, api = _channel(api=_Api(fail_on_call=2))

    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        await channel.send(OutboundMessage(channel="qq", chat_id="G1", content="z" * 5_000))

    assert len(api.calls) == 2
    assert "Error sending QQ message" in caplog.text


@pytest.mark.asyncio
async def test_empty_content_sends_nothing(caplog):
    channel, api = _channel()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        await channel.send(OutboundMessage(channel="qq", chat_id="G1", content=""))

    assert api.calls == []
    assert "empty message" in caplog.text


@pytest.mark.asyncio
async def test_attachments_are_dropped_loudly(caplog):
    """Group upload takes a public URL and botpy exposes no byte upload for
    groups, so an attachment cannot be delivered -- and the sender believes it
    was."""
    channel, api = _channel()

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        await channel.send(
            OutboundMessage(
                channel="qq",
                chat_id="G1",
                content="看图",
                media=["/workspace/attachments/qq/a.jpg"],
            )
        )

    assert [call["content"] for call in api.calls] == ["看图"]
    assert "cannot send attachments" in caplog.text


@pytest.mark.asyncio
async def test_without_a_client_nothing_is_sent(caplog):
    channel = QQChannel(QQConfig(app_id="1", app_secret="s"), MessageBus())

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        await channel.send(OutboundMessage(channel="qq", chat_id="G1", content="你好"))

    assert "client not initialized" in caplog.text
