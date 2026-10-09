"""The QQ reconnect loop and its logging.

No botpy import anywhere here: ``_run_bot`` only touches ``self._client`` and
``self._running``, so a stub client is enough to exercise the whole loop.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl import qq as qq_module
from openharness.channels.impl.qq import QQChannel
from openharness.config.schema import QQConfig

_LOGGER = "openharness.channels.impl.qq"


class _StubClient:
    """Client whose ``start()`` follows a scripted list of behaviours."""

    def __init__(self, channel: QQChannel, behaviours: list) -> None:
        self._channel = channel
        self._behaviours = behaviours
        self.calls = 0

    async def start(self, **kwargs) -> None:
        del kwargs
        behaviour = self._behaviours[min(self.calls, len(self._behaviours) - 1)]
        self.calls += 1
        await behaviour(self)


def _channel() -> QQChannel:
    return QQChannel(QQConfig(app_id="1", app_secret="s"), MessageBus())


@pytest.mark.asyncio
async def test_sdk_internal_timeout_is_not_reported_as_a_wedge(monkeypatch, caplog):
    channel = _channel()

    async def timed_out(client: _StubClient) -> None:
        client._channel._running = False
        raise TimeoutError("botpy gave up")

    channel._client = _StubClient(channel, [timed_out])
    channel._running = True

    with caplog.at_level(logging.DEBUG, logger=_LOGGER):
        await channel._run_bot()

    # Since 3.11 asyncio.TimeoutError *is* TimeoutError, so this used to be
    # logged as a 15-minute wedge and send us chasing a hang that never existed.
    assert "timed out after" in caplog.text
    assert "wedged" not in caplog.text


@pytest.mark.asyncio
async def test_our_own_bound_is_reported_as_a_wedge(monkeypatch, caplog):
    channel = _channel()
    monkeypatch.setattr(qq_module, "_CONNECT_TIMEOUT_SECONDS", 0.05)

    async def hangs(client: _StubClient) -> None:
        await asyncio.sleep(3600)

    client = _StubClient(channel, [hangs])
    channel._client = client
    channel._running = True
    caplog.set_level(logging.DEBUG, logger=_LOGGER)

    task = asyncio.create_task(channel._run_bot())
    try:
        for _ in range(200):
            if "wedged" in caplog.text:
                break
            await asyncio.sleep(0.01)
        channel._running = False
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert client.calls == 1
    assert "wedged for" in caplog.text


@pytest.mark.asyncio
async def test_reconnect_backoff_doubles_up_to_the_ceiling(monkeypatch):
    channel = _channel()
    monkeypatch.setattr(qq_module, "_RECONNECT_BASE_SECONDS", 1)
    monkeypatch.setattr(qq_module, "_RECONNECT_MAX_SECONDS", 4)

    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def fail(client: _StubClient) -> None:
        if client.calls >= 5:
            client._channel._running = False
        raise RuntimeError("connection refused")

    channel._client = _StubClient(channel, [fail])
    channel._running = True

    await channel._run_bot()

    assert delays == [1, 2, 4, 4]


@pytest.mark.asyncio
async def test_a_successful_run_resets_the_backoff(monkeypatch):
    channel = _channel()
    monkeypatch.setattr(qq_module, "_RECONNECT_BASE_SECONDS", 1)
    monkeypatch.setattr(qq_module, "_RECONNECT_MAX_SECONDS", 64)

    delays: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        delays.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    async def fail(client: _StubClient) -> None:
        raise RuntimeError("connection refused")

    async def succeed(client: _StubClient) -> None:
        # A clean return means the socket closed after a healthy session.
        return None

    async def stop(client: _StubClient) -> None:
        client._channel._running = False
        raise RuntimeError("connection refused")

    channel._client = _StubClient(channel, [fail, fail, succeed, stop])
    channel._running = True

    await channel._run_bot()

    # 1 then 2 while failing, back to 1 after the healthy run, then stopped.
    assert delays == [1, 2, 1]
