"""Inbound QQ image downloads.

QQ signs its media URLs with a short-lived ``rkey``, so the fetch has to happen
inside the event that carried the message. These tests pin the bounds and the
"never let a CDN failure lose the message" behaviour.
"""

from __future__ import annotations

import json
import logging

import pytest

from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl import qq as qq_module
from openharness.channels.impl.qq import QQChannel
from openharness.channels.impl.qq_rich import MediaRef
from openharness.config.schema import QQConfig

_LOGGER = "openharness.channels.impl.qq"
_URL = "https://multimedia.nt.qq.com.cn/download?fileid=ABC&rkey=SECRET"

# A 1x1 PNG. Real magic bytes on purpose: the channel now refuses to write
# anything it cannot identify as an image from the bytes alone, so the
# stand-in text these tests used to use would simply be skipped.
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"\x90wS\xde\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01"
    b"\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
)

# RIFF....WEBP -- a valid image whose declared filename says otherwise.
_WEBP_BYTES = b"RIFF\x24\x00\x00\x00WEBPVP8 \x18\x00\x00\x00" + b"\x00" * 24

# What a stale ``rkey`` actually returns: HTTP 200 with an HTML body.
_HTML_BYTES = b"<!DOCTYPE html><html><body>404</body></html>"


class _Response:
    def __init__(
        self,
        content: bytes,
        status_error: Exception | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.content = content
        self.headers = headers or {}
        self._status_error = status_error

    def raise_for_status(self) -> None:
        if self._status_error is not None:
            raise self._status_error


class _StubHttp:
    def __init__(self, responses: dict[str, object]) -> None:
        self._responses = responses
        self.requested: list[str] = []

    async def get(self, url: str) -> _Response:
        self.requested.append(url)
        outcome = self._responses.get(url, _Response(b""))
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def _media_root(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENHARNESS_CHANNEL_MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.delenv("OHMO_WORKSPACE", raising=False)
    return tmp_path / "media" / "qq"


def _channel(http: _StubHttp) -> QQChannel:
    channel = QQChannel(QQConfig(app_id="1", app_secret="s"), MessageBus())
    channel._http = http
    return channel


@pytest.mark.asyncio
async def test_download_writes_the_file_and_maps_the_url(_media_root):
    http = _StubHttp({_URL: _Response(_PNG_BYTES)})
    channel = _channel(http)

    saved = await channel._download_media("MSG1", (MediaRef(url=_URL, filename="a.jpg"),))

    path = _media_root / "MSG1_1_a.png"
    assert saved == {_URL: str(path)}
    assert path.read_bytes() == _PNG_BYTES


@pytest.mark.asyncio
async def test_saved_extension_follows_the_bytes_not_the_filename(_media_root):
    """QQ called it a .jpg; the bytes are a WebP, and that is what is stored."""
    http = _StubHttp({_URL: _Response(_WEBP_BYTES)})
    channel = _channel(http)

    saved = await channel._download_media("MSG1", (MediaRef(url=_URL, filename="a.jpg"),))

    path = _media_root / "MSG1_1_a.webp"
    assert saved == {_URL: str(path)}
    assert not (_media_root / "MSG1_1_a.jpg").exists()


@pytest.mark.asyncio
async def test_a_non_image_body_is_never_written(_media_root, caplog):
    """An expired rkey answers 200 with HTML. Writing it would poison the turn."""
    http = _StubHttp({_URL: _Response(_HTML_BYTES, headers={"content-type": "text/html"})})
    channel = _channel(http)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        saved = await channel._download_media("MSG1", (MediaRef(url=_URL, filename="a.jpg"),))

    assert saved == {}
    assert not (_media_root / "MSG1_1_a.jpg").exists()
    assert not (_media_root / "MSG1_1_a.png").exists()
    assert "not a supported image" in caplog.text
    # The log has to identify the cause without a packet capture.
    assert "text/html" in caplog.text


@pytest.mark.asyncio
async def test_one_failed_image_does_not_lose_the_others(_media_root, caplog):
    ok = _URL + "-ok"
    bad = _URL + "-bad"
    http = _StubHttp({ok: _Response(_PNG_BYTES), bad: RuntimeError("connection reset")})
    channel = _channel(http)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        saved = await channel._download_media(
            "MSG1",
            (MediaRef(url=bad, filename="bad.jpg"), MediaRef(url=ok, filename="ok.jpg")),
        )

    assert list(saved) == [ok]
    assert "download failed" in caplog.text


@pytest.mark.asyncio
async def test_total_budget_caps_across_images_not_per_file(_media_root, caplog, monkeypatch):
    monkeypatch.setattr(qq_module, "MAX_TOTAL_MEDIA_BYTES", len(_PNG_BYTES) + 1)
    first = _URL + "-1"
    second = _URL + "-2"
    http = _StubHttp({first: _Response(_PNG_BYTES), second: _Response(_PNG_BYTES)})
    channel = _channel(http)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        saved = await channel._download_media(
            "MSG1",
            (MediaRef(url=first, filename="1.png"), MediaRef(url=second, filename="2.png")),
        )

    # Each file is far under MAX_ATTACHMENT_BYTES; only their sum breaks the cap.
    assert list(saved) == [first]
    assert "over the message budget" in caplog.text


@pytest.mark.asyncio
async def test_an_http_error_status_is_treated_as_a_failure(_media_root, caplog):
    http = _StubHttp({_URL: _Response(b"", status_error=RuntimeError("403"))})
    channel = _channel(http)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        saved = await channel._download_media("MSG1", (MediaRef(url=_URL, filename="a.jpg"),))

    assert saved == {}
    assert "download failed" in caplog.text


@pytest.mark.asyncio
async def test_oversized_attachments_are_skipped(_media_root, caplog, monkeypatch):
    monkeypatch.setattr(qq_module, "MAX_ATTACHMENT_BYTES", 4)
    http = _StubHttp({_URL: _Response(b"way too many bytes")})
    channel = _channel(http)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        saved = await channel._download_media("MSG1", (MediaRef(url=_URL, filename="a.jpg"),))

    assert saved == {}
    assert "size cap" in caplog.text
    assert not (_media_root / "MSG1_1_a.jpg").exists()


@pytest.mark.asyncio
async def test_only_the_first_images_are_fetched(_media_root, caplog, monkeypatch):
    monkeypatch.setattr(qq_module, "MAX_IMAGES_PER_MESSAGE", 2)
    urls = [f"{_URL}-{index}" for index in range(5)]
    http = _StubHttp({url: _Response(_PNG_BYTES) for url in urls})
    channel = _channel(http)

    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        saved = await channel._download_media(
            "MSG1", tuple(MediaRef(url=url, filename=f"{index}.jpg") for index, url in enumerate(urls))
        )

    assert http.requested == urls[:2]
    assert len(saved) == 2
    assert "only the first" in caplog.text


@pytest.mark.asyncio
async def test_no_refs_opens_no_connection():
    class _NeverUsed:
        async def get(self, url: str) -> _Response:
            raise AssertionError("no image means no request")

    channel = QQChannel(QQConfig(app_id="1", app_secret="s"), MessageBus())
    channel._http = _NeverUsed()  # type: ignore[assignment]

    assert await channel._download_media("MSG1", ()) == {}


_IMAGE_TRANSCRIPT = (
    "=== 消息 1 ===\n"
    "[消息内容] 呼哈哈哈哈！\n"
    "\n"
    "=== 消息 2 ===\n"
    f"[附件1] 类型:图片 文件名:a.jpg 尺寸:261x264 大小:8.8KB URL:{_URL}\n"
    "\n"
    "=== 消息 3 ===\n"
    "[消息内容] 别急，我在调试"
)


class _FakeMessage:
    """The subset of a botpy GroupMessage this channel actually reads."""

    def __init__(self, raw: dict) -> None:
        self.id = "MSG1"
        self.group_openid = "G1"
        self.content = "@奇希莉卡 看看这个"
        self.author = type("_Author", (), {"member_openid": "U1"})()
        self.raw = raw


@pytest.mark.asyncio
async def test_group_message_carries_local_paths_instead_of_expiring_urls(_media_root):
    http = _StubHttp({_URL: _Response(_PNG_BYTES)})
    bus = MessageBus()
    channel = QQChannel(QQConfig(app_id="1", app_secret="s", allow_from=["*"]), bus)
    channel._http = http

    await channel._on_group_message(
        _FakeMessage(
            {
                "id": "MSG1",
                "timestamp": "2026-10-09T10:38:56+08:00",
                "author": {"member_openid": "U1", "username": "调皮的小男孩"},
                "msg_elements": [{"content": _IMAGE_TRANSCRIPT}],
            }
        )
    )

    inbound = await bus.consume_inbound()
    path = str(_media_root / "MSG1_1_a.png")
    assert inbound.media == [path]
    assert path in inbound.content
    # The rkey dies within minutes; it must not reach the model or the archive.
    # `qq_segments` matters as much as `content`: the gossip tool archives one of
    # them verbatim, so a live URL there lands in the store untouched.
    assert "rkey" not in inbound.content
    assert "SECRET" not in json.dumps(inbound.metadata, ensure_ascii=False)
    # The @-mention is stripped, and the transcript is marked as background
    # rather than handed over as if the sender had said it.
    assert inbound.content.startswith("看看这个\n\n[群聊转写 · ")
    assert inbound.metadata["sender_display_name"] == "调皮的小男孩"
    segments = inbound.metadata["qq_segments"]
    assert [segment["index"] for segment in segments] == [1, 2, 3]
    assert path in segments[1]["content"]
    # The marker is a message-layer annotation only. A segment is archived word
    # for word, so a marker reaching one would be filed away as content.
    assert all("群聊转写" not in segment["content"] for segment in segments)


@pytest.mark.asyncio
async def test_group_message_survives_a_failed_download(_media_root):
    http = _StubHttp({_URL: RuntimeError("connection reset")})
    bus = MessageBus()
    channel = QQChannel(QQConfig(app_id="1", app_secret="s", allow_from=["*"]), bus)
    channel._http = http

    await channel._on_group_message(
        _FakeMessage(
            {
                "id": "MSG1",
                "author": {"member_openid": "U1"},
                "msg_elements": [{"content": _IMAGE_TRANSCRIPT}],
            }
        )
    )

    inbound = await bus.consume_inbound()
    assert inbound.media == []
    assert "[图片下载失败]" in inbound.content


@pytest.mark.asyncio
async def test_stop_closes_the_http_client():
    closed: list[bool] = []

    class _Closable(_StubHttp):
        async def aclose(self) -> None:
            closed.append(True)

    channel = QQChannel(QQConfig(app_id="1", app_secret="s"), MessageBus())
    channel._http = _Closable({})  # type: ignore[assignment]

    await channel.stop()

    assert closed == [True]
    assert channel._http is None
