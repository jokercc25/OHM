"""QQ channel implementation using botpy SDK."""

import asyncio
import json
import logging
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from openharness.channels.bus.events import OutboundMessage
from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.base import BaseChannel, resolve_channel_media_dir
from openharness.channels.impl.qq_rich import (
    MediaRef,
    census,
    flatten_group_payload,
    rewrite_media_links,
    safe_attachment_name,
    split_transcript,
)
from openharness.config.schema import QQConfig
from openharness.utils.fs import atomic_write_text
from openharness.utils.helpers import image_extension_for, sniff_image_type, split_message

logger = logging.getLogger(__name__)

try:
    import botpy
    from botpy.message import C2CMessage, GroupMessage

    QQ_AVAILABLE = True
except ImportError:
    QQ_AVAILABLE = False
    botpy = None
    C2CMessage = None
    GroupMessage = None

if TYPE_CHECKING:
    from botpy.message import C2CMessage, GroupMessage

# Best-effort strip of a leading @-mention. Official group messages fire only
# when the bot is @mentioned, and some payloads keep the @ text in content.
_AT_MENTION_RE = re.compile(r"^\s*@[^\s]+\s*")

# Subdirectory of the workspace log dir used by the OHMO_QQ_DEBUG_PAYLOAD dump.
_DEBUG_PAYLOAD_DIRNAME = "qq-payload"

# How long a single client.start() may run before we treat the connection as
# wedged. botpy can hang here without raising, which is how the bot went silent
# for two weeks: the process never exited, so `restart: unless-stopped` never
# fired. Bounding it forces the reconnect loop to make progress.
_CONNECT_TIMEOUT_SECONDS = 15 * 60
_RECONNECT_BASE_SECONDS = 5
_RECONNECT_MAX_SECONDS = 300

# Attachment download bounds. QQ hands out rkey-signed media URLs that expire
# within minutes, so the only chance to keep an image is to fetch it during the
# event that carried it.
MEDIA_DOWNLOAD_TIMEOUT_SECONDS = 30.0
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024
MAX_IMAGES_PER_MESSAGE = 6
# Per-message budget across all images, not per file. Every image is inlined as
# base64 into one request body, and providers cap that body (DeepSeek: 48 MiB,
# with a 32 MiB ceiling per image). Six images at the per-file cap would be
# 120 MiB of raw bytes -- far past the limit -- so the per-file bound alone does
# not bound anything.
MAX_TOTAL_MEDIA_BYTES = 24 * 1024 * 1024

# QQ rejects a single text message longer than 2000 characters (error 40054007)
# rather than truncating it, so anything longer has to go out as several
# messages. Nothing splits before this point, which meant one long reply was
# simply lost with an error in the log.
QQ_MAX_MESSAGE_LEN = 2000


def _install_botpy_payload_hook() -> None:
    """Keep the raw QQ event payload on the message object.

    botpy reads a fixed set of keys out of the event and drops the dict, so
    fields like ``msg_elements`` / ``ark_data`` / ``message_scene`` and the
    author's nickname are unreachable. ``connection.parse_*`` looks the class up
    in its module globals at call time, and ``__slots__`` chains are add-only,
    so rebinding with a subclass that stashes ``data`` recovers them.

    Feature-detected and non-fatal: if botpy changes shape we degrade to the
    stock fields rather than crashing the channel.
    """
    try:
        import botpy.connection as _bc
    except Exception:
        logger.exception("QQ raw payload hook unavailable; rich fields will be skipped")
        return
    if getattr(getattr(_bc, "GroupMessage", None), "_ohmo_raw_capable", False):
        return
    try:

        class _RawGroupMessage(_bc.GroupMessage):  # type: ignore[misc, valid-type]
            __slots__ = ("raw",)
            _ohmo_raw_capable = True

            def __init__(self, api, event_id, data):
                super().__init__(api, event_id, data)
                self.raw = dict(data or {})

        class _RawC2CMessage(_bc.C2CMessage):  # type: ignore[misc, valid-type]
            __slots__ = ("raw",)
            _ohmo_raw_capable = True

            def __init__(self, api, event_id, data):
                super().__init__(api, event_id, data)
                self.raw = dict(data or {})

        _bc.GroupMessage = _RawGroupMessage
        _bc.C2CMessage = _RawC2CMessage
        logger.info("QQ raw payload hook installed")
    except Exception:
        logger.exception("QQ raw payload hook unavailable; rich fields will be skipped")


def _debug_payload_enabled() -> bool:
    """Whether the Phase-0 payload census dump is switched on."""
    return os.environ.get("OHMO_QQ_DEBUG_PAYLOAD", "").strip().lower() in {"1", "true", "yes", "on"}


def _dump_qq_payload(content, message_id: str) -> None:
    """Persist one redacted raw payload for offline inspection.

    ``content`` is a :class:`RichContent` whose ``raw`` field is already
    redacted (rkey / auth_token / msg_idx removed) by ``qq_rich``.
    """
    raw = getattr(content, "raw", None)
    if not isinstance(raw, dict):
        return
    workspace = os.environ.get("OHMO_WORKSPACE") or "."
    dump_dir = Path(workspace) / "logs" / _DEBUG_PAYLOAD_DIRNAME
    try:
        dump_dir.mkdir(parents=True, exist_ok=True)
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", message_id or "unknown")[:80] or "unknown"
        atomic_write_text(dump_dir / f"{name}.json", json.dumps(raw, ensure_ascii=False, indent=2))
    except Exception:
        logger.exception("QQ payload dump failed")


def _make_bot_class(channel: "QQChannel") -> "type[botpy.Client]":
    """Create a botpy Client subclass bound to the given channel."""
    intents = botpy.Intents(public_messages=True, direct_message=True)

    class _Bot(botpy.Client):
        def __init__(self):
            # Disable botpy's file log — not using loguru; default "botpy.log" fails on read-only fs
            super().__init__(intents=intents, ext_handlers=False)

        async def on_ready(self):
            logger.info("QQ bot ready: %s", self.robot.name)

        async def on_c2c_message_create(self, message: "C2CMessage"):
            await channel._on_message(message)

        async def on_direct_message_create(self, message):
            await channel._on_message(message)

        async def on_group_at_message_create(self, message: "GroupMessage"):
            await channel._on_group_message(message)

    return _Bot


class QQChannel(BaseChannel):
    """QQ channel using botpy SDK with WebSocket connection."""

    name = "qq"

    def __init__(self, config: QQConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: QQConfig = config
        self._client: "botpy.Client | None" = None
        self._processed_ids: deque = deque(maxlen=1000)
        self._msg_seq: int = 1  # 消息序列号，避免被 QQ API 去重
        # chat_id for a group message is the group_openid; track which chat_ids
        # are groups so send() can route to the right post_* API.
        self._group_openids: set[str] = set()
        # Last msg_id seen per group, required for passive group replies.
        self._group_msg_ids: dict[str, str] = {}
        # Created on first use so a channel that never sees an image never opens
        # a connection pool; closed in stop().
        self._http: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Start the QQ bot."""
        if not QQ_AVAILABLE:
            logger.error("QQ SDK not installed. Run: pip install qq-botpy")
            return

        if not self.config.app_id or not self.config.app_secret:
            logger.error("QQ app_id and app_secret not configured")
            return

        _install_botpy_payload_hook()
        self._running = True
        BotClass = _make_bot_class(self)
        self._client = BotClass()

        logger.info("QQ bot started (C2C private message + group @mention)")
        await self._run_bot()

    async def _run_bot(self) -> None:
        """Run the bot connection with auto-reconnect and backoff."""
        backoff = _RECONNECT_BASE_SECONDS
        while self._running:
            started = time.monotonic()
            try:
                await asyncio.wait_for(
                    self._client.start(appid=self.config.app_id, secret=self.config.app_secret),
                    timeout=_CONNECT_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                # Since Python 3.11 asyncio.TimeoutError *is* the builtin
                # TimeoutError, so botpy's own internal timeouts land here too.
                # Measuring elapsed time is what separates "our bound fired" from
                # "the SDK gave up quickly" -- without it every SDK timeout got
                # reported as a 15-minute wedge, which sent us hunting for a hang
                # that was never there.
                elapsed = time.monotonic() - started
                if elapsed >= _CONNECT_TIMEOUT_SECONDS * 0.9:
                    # botpy can wedge without raising; without this bound the
                    # process never exits and `restart: unless-stopped` never fires.
                    logger.error(
                        "QQ bot connection wedged for %.0fs, forcing reconnect", elapsed
                    )
                else:
                    logger.warning(
                        "QQ bot connection timed out after %.0fs (inside the SDK), reconnecting",
                        elapsed,
                    )
            except Exception as e:
                logger.warning("QQ bot error: %s", e)
            else:
                backoff = _RECONNECT_BASE_SECONDS
            if self._running:
                logger.info("Reconnecting QQ bot in %s seconds...", backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _RECONNECT_MAX_SECONDS)

    async def stop(self) -> None:
        """Stop the QQ bot."""
        self._running = False
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception:
                pass
            self._http = None
        logger.info("QQ bot stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through QQ (C2C or group).

        A message over ``QQ_MAX_MESSAGE_LEN`` is split rather than sent: QQ
        rejects the whole thing otherwise, and a caller that composed a long
        reply -- a verbatim archive dump, say -- has already decided the content
        matters. ``_msg_seq`` increments per chunk, which is also what stops the
        platform from de-duplicating them into one.
        """
        if not self._client:
            logger.warning("QQ client not initialized")
            return

        chunks = split_message(msg.content or "", QQ_MAX_MESSAGE_LEN)
        if not chunks:
            logger.warning("QQ channel asked to send an empty message chat_id=%s", msg.chat_id)
            return
        if msg.media:
            # Group image upload takes a public URL and botpy exposes no byte
            # upload for groups, so an attachment cannot be delivered here.
            # Logged rather than silent because the sender believes it went.
            logger.warning(
                "QQ channel cannot send attachments, dropped count=%s chat_id=%s",
                len(msg.media),
                msg.chat_id,
            )

        msg_id = msg.metadata.get("message_id")
        is_group = msg.chat_id in self._group_openids
        # Group replies are passive: must attach the msg_id of the @-message we
        # are responding to.
        group_msg_id = msg_id or self._group_msg_ids.get(msg.chat_id)

        for chunk in chunks:
            self._msg_seq += 1  # 递增序列号
            try:
                if is_group:
                    await self._client.api.post_group_message(
                        group_openid=msg.chat_id,
                        msg_type=0,
                        content=chunk,
                        msg_id=group_msg_id,
                        msg_seq=self._msg_seq,
                    )
                else:
                    await self._client.api.post_c2c_message(
                        openid=msg.chat_id,
                        msg_type=0,
                        content=chunk,
                        msg_id=msg_id,
                        msg_seq=self._msg_seq,  # 添加序列号避免去重
                    )
            except Exception as e:
                # Stop rather than plough on: the usual cause is the five-reply
                # budget for one inbound message, which fails every later chunk
                # too, so continuing only buys noise.
                logger.error("Error sending QQ message: %s", e)
                return

    def _http_client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=MEDIA_DOWNLOAD_TIMEOUT_SECONDS,
                follow_redirects=True,
            )
        return self._http

    async def _download_media(self, message_id: str, refs: tuple[MediaRef, ...]) -> dict[str, str]:
        """Fetch this message's images; return the URL -> local path map.

        QQ's media URLs carry a short-lived ``rkey``, so these must be fetched
        now or never -- the URL is deliberately never persisted. Every failure
        is swallowed into a log line: a member forwarding an image must not
        lose the rest of the message because the CDN was slow.

        Bytes are sniffed before anything is written. A CDN that answers with an
        HTML error page -- or with an image in a format the vision providers do
        not accept -- must not land on disk under a ``.jpg`` name: the file is
        inlined into the request on this turn and, because the block is
        persisted, on every later turn too. One bad file written here is a
        conversation the provider rejects forever after.
        """
        if not refs:
            return {}
        media_dir = resolve_channel_media_dir(self.name)
        saved: dict[str, str] = {}
        total_bytes = 0
        rejected_format = 0
        rejected_budget = 0
        for index, ref in enumerate(refs[:MAX_IMAGES_PER_MESSAGE], start=1):
            try:
                response = await self._http_client().get(ref.url)
                response.raise_for_status()
                content = response.content
                if len(content) > MAX_ATTACHMENT_BYTES:
                    logger.warning(
                        "QQ attachment over the size cap, skipped bytes=%s message_id=%s",
                        len(content),
                        message_id,
                    )
                    continue
                media_type = sniff_image_type(content)
                if media_type is None:
                    rejected_format += 1
                    # What the CDN *claimed* plus the first bytes of what it
                    # actually sent -- that pair is what identifies the cause
                    # (an HTML error page, a truncated body, a format the
                    # providers reject) without a packet capture.
                    headers = getattr(response, "headers", None)
                    declared = headers.get("content-type") if hasattr(headers, "get") else None
                    logger.warning(
                        "QQ attachment is not a supported image, skipped bytes=%s "
                        "content_type=%r head=%s message_id=%s",
                        len(content),
                        declared,
                        content[:16].hex(),
                        message_id,
                    )
                    continue
                if total_bytes + len(content) > MAX_TOTAL_MEDIA_BYTES:
                    rejected_budget += 1
                    logger.warning(
                        "QQ attachment over the message budget, skipped bytes=%s kept=%s "
                        "message_id=%s",
                        len(content),
                        total_bytes,
                        message_id,
                    )
                    continue
                # Name the file after the sniffed format rather than after what
                # QQ called it: the extension is what drives the media type on
                # the way back out, so an honest one keeps that truthful.
                stem = Path(safe_attachment_name(message_id, index, ref.filename)).stem
                target = media_dir / f"{stem}{image_extension_for(media_type)}"
                target.write_bytes(content)
                total_bytes += len(content)
                saved[ref.url] = str(target)
            except Exception as e:
                logger.warning("QQ attachment download failed message_id=%s error=%s", message_id, e)
        if len(refs) > MAX_IMAGES_PER_MESSAGE:
            logger.warning(
                "QQ message carried %s images, only the first %s were kept message_id=%s",
                len(refs),
                MAX_IMAGES_PER_MESSAGE,
                message_id,
            )
        if rejected_format or rejected_budget:
            logger.warning(
                "QQ message kept %s of %s images (unsupported_format=%s over_budget=%s) message_id=%s",
                len(saved),
                len(refs),
                rejected_format,
                rejected_budget,
                message_id,
            )
        return saved

    async def _on_message(self, data: "C2CMessage") -> None:
        """Handle incoming C2C (private) message from QQ."""
        try:
            # Dedup by message ID
            if data.id in self._processed_ids:
                return
            self._processed_ids.append(data.id)

            author = data.author
            user_id = str(getattr(author, 'id', None) or getattr(author, 'user_openid', 'unknown'))
            content = (data.content or "").strip()
            if not content:
                return

            await self._handle_message(
                sender_id=user_id,
                chat_id=user_id,
                content=content,
                metadata={"message_id": data.id},
            )
        except Exception:
            logger.exception("Error handling QQ message")

    async def _on_group_message(self, data: "GroupMessage") -> None:
        """Handle an incoming group @-mention message from QQ."""
        try:
            # botpy's GroupMessage exposes the message id as `id` (there is no
            # separate `msg_id` attribute); `id` is also what passive group
            # replies must echo back as msg_id.
            message_id = str(getattr(data, "id", None) or "")
            if message_id and message_id in self._processed_ids:
                return
            if message_id:
                self._processed_ids.append(message_id)

            group_openid = str(getattr(data, "group_openid", "") or "")
            if not group_openid:
                logger.warning("QQ group message missing group_openid, ignoring")
                return

            # Raw payload is only present when _install_botpy_payload_hook()
            # succeeded; without it we fall back to the stock botpy fields.
            raw = getattr(data, "raw", None)
            if not isinstance(raw, dict):
                raw = None
            rich = flatten_group_payload(raw)

            # Phase-0 instrument. Runs *before* the content checks below so that
            # card-only messages (empty `content`) are still observed — those are
            # exactly the messages we need to characterise.
            if raw is not None and _debug_payload_enabled():
                logger.info("QQ payload census %s", census(raw))
                _dump_qq_payload(rich, message_id)

            author = data.author
            member_id = str(
                getattr(author, "member_openid", None)
                or getattr(author, "id", None)
                or "unknown"
            )

            # The @-message carries two things: what the sender typed, and QQ's
            # transcript of the recent messages. A forwarded card arrives in the
            # transcript, so an empty `content` must NOT be treated as an empty
            # message -- that was dropping exactly the messages we want.
            typed = _AT_MENTION_RE.sub("", (data.content or "").strip(), count=1).strip()
            transcript = rich.text
            if not typed and not transcript:
                return

            # Must happen here, in the event that carried the message: the URLs
            # are rkey-signed and stop working within minutes. The transcript is
            # rewritten in place so what reaches the archive -- and the model --
            # points at a file instead of a link that is already expiring.
            saved = await self._download_media(message_id, rich.media)
            media_paths = list(saved.values())
            if rich.media:
                transcript = rewrite_media_links(transcript, saved)

            content = f"{typed}\n\n{transcript}" if typed and transcript else (typed or transcript)
            # Segments are re-split from the *rewritten* transcript: the gossip
            # tool archives a segment verbatim, so a stale rkey URL reaching
            # `metadata["qq_segments"]` would be written straight into the store.
            segments = split_transcript(transcript)

            # Remember this chat as a group so send() routes to post_group_message,
            # and stash the message id so passive group replies can echo msg_id.
            self._group_openids.add(group_openid)
            if message_id:
                self._group_msg_ids[group_openid] = message_id

            # botpy's GroupMessage._User only carries member_openid, so the
            # nickname is reachable only through the raw payload. Supplying it
            # lets runtime._build_speaker_context name the sender instead of
            # printing the opaque "group_openid|member_openid" pair.
            author_raw = raw.get("author") if raw else None
            display_name = ""
            if isinstance(author_raw, dict):
                display_name = str(author_raw.get("username") or "").strip()

            metadata = {
                "message_id": message_id,
                "group_openid": group_openid,
                "chat_type": "group",
                "is_group": True,
                "sender_member_id": member_id,
            }
            if display_name:
                metadata["sender_display_name"] = display_name
            if raw is not None:
                metadata["qq_timestamp"] = raw.get("timestamp")
                metadata["qq_message_ids"] = list(rich.message_ids)
                metadata["qq_payload"] = rich.raw
            if segments:
                # Verbatim section bodies, keyed by QQ's own numbering, so the
                # gossip tool can archive one of them without the model ever
                # supplying the text.
                metadata["qq_segments"] = [
                    {"index": segment.index, "content": segment.content}
                    for segment in segments
                ]

            # Pipe-join group + member so allow_from can match either the whole
            # group (group_openid) or a specific member (member_openid).
            await self._handle_message(
                sender_id=f"{group_openid}|{member_id}",
                chat_id=group_openid,
                content=content,
                media=media_paths,
                metadata=metadata,
            )
        except Exception:
            logger.exception("Error handling QQ group message")
