"""QQ channel implementation using botpy SDK."""

import asyncio
import logging
import re
from collections import deque
from typing import TYPE_CHECKING


from openharness.channels.bus.events import OutboundMessage
from openharness.channels.bus.queue import MessageBus
from openharness.channels.impl.base import BaseChannel
from openharness.config.schema import QQConfig

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

    async def start(self) -> None:
        """Start the QQ bot."""
        if not QQ_AVAILABLE:
            logger.error("QQ SDK not installed. Run: pip install qq-botpy")
            return

        if not self.config.app_id or not self.config.app_secret:
            logger.error("QQ app_id and app_secret not configured")
            return

        self._running = True
        BotClass = _make_bot_class(self)
        self._client = BotClass()

        logger.info("QQ bot started (C2C private message + group @mention)")
        await self._run_bot()

    async def _run_bot(self) -> None:
        """Run the bot connection with auto-reconnect."""
        while self._running:
            try:
                await self._client.start(appid=self.config.app_id, secret=self.config.app_secret)
            except Exception as e:
                logger.warning("QQ bot error: %s", e)
            if self._running:
                logger.info("Reconnecting QQ bot in 5 seconds...")
                await asyncio.sleep(5)

    async def stop(self) -> None:
        """Stop the QQ bot."""
        self._running = False
        if self._client:
            try:
                await self._client.close()
            except Exception:
                pass
        logger.info("QQ bot stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through QQ (C2C or group)."""
        if not self._client:
            logger.warning("QQ client not initialized")
            return
        try:
            self._msg_seq += 1  # 递增序列号
            msg_id = msg.metadata.get("message_id")
            if msg.chat_id in self._group_openids:
                # Group replies are passive: must attach the msg_id of the
                # @-message we are responding to.
                group_msg_id = msg_id or self._group_msg_ids.get(msg.chat_id)
                await self._client.api.post_group_message(
                    group_openid=msg.chat_id,
                    msg_type=0,
                    content=msg.content,
                    msg_id=group_msg_id,
                    msg_seq=self._msg_seq,
                )
            else:
                await self._client.api.post_c2c_message(
                    openid=msg.chat_id,
                    msg_type=0,
                    content=msg.content,
                    msg_id=msg_id,
                    msg_seq=self._msg_seq,  # 添加序列号避免去重
                )
        except Exception as e:
            logger.error("Error sending QQ message: %s", e)

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

            author = data.author
            member_id = str(
                getattr(author, "member_openid", None)
                or getattr(author, "id", None)
                or "unknown"
            )

            content = (data.content or "").strip()
            if not content:
                return
            content = _AT_MENTION_RE.sub("", content, count=1).strip()
            if not content:
                return

            # Remember this chat as a group so send() routes to post_group_message,
            # and stash the message id so passive group replies can echo msg_id.
            self._group_openids.add(group_openid)
            if message_id:
                self._group_msg_ids[group_openid] = message_id

            # Pipe-join group + member so allow_from can match either the whole
            # group (group_openid) or a specific member (member_openid).
            await self._handle_message(
                sender_id=f"{group_openid}|{member_id}",
                chat_id=group_openid,
                content=content,
                metadata={
                    "message_id": message_id,
                    "group_openid": group_openid,
                    "chat_type": "group",
                    "is_group": True,
                    "sender_member_id": member_id,
                },
            )
        except Exception:
            logger.exception("Error handling QQ group message")
