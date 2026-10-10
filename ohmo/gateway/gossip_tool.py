"""ohmo-only gossip archive tools.

The trigger is the tool description itself -- there is no keyword list and no
regex. The model reads the message and decides whether it is worth keeping, and
later whether the group is asking to see what is in there.

Reading is split from showing on purpose. ``ohmo_list_gossip`` and
``ohmo_show_gossip`` publish the group's own archive text themselves, through a
callable the gateway injects, instead of handing it back as tool output for the
model to retype. A model that retypes is a model that can drop a line, fix a
typo, or summarise -- and the archive is only worth keeping if what comes back
out is what went in.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from openharness.memory.types import MemoryHeader
from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

from ohmo.gossip import (
    find_gossip_entry,
    format_gossip_body,
    format_gossip_listing,
    format_gossip_messages,
    list_gossip_headers,
    load_gossip_entry,
    save_gossip_entry,
    search_gossip,
)

logger = logging.getLogger(__name__)

# Injected by the gateway, mirroring PublishGroupWelcome: the tool composes the
# text, the service owns the bus. Arguments are (chat_id, content, channel).
PublishGossipText = Callable[[str, str, str], Awaitable[None] | None]

# How many entries a listing shows when the model does not say.
_DEFAULT_LIST_LIMIT = 20
_MAX_LIST_LIMIT = 50



class OhmoSaveGossipInput(BaseModel):
    """Arguments selected by the model for archiving one piece of gossip."""

    gist: str = Field(
        description="一句中文主旨（不超过 40 字），写清“谁、什么事”。这是别人以后检索的入口。"
    )
    which: int | None = Field(
        default=None,
        description=(
            "要存档的是群聊转写里第几条，即 “=== 消息 N ===” 的 N。"
            "只存其中一条时必填；省略表示存档整段转写。"
        ),
    )
    title: str | None = Field(
        default=None,
        description="可选短标题（不超过 20 字）。省略则直接用主旨当标题。",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="可选标签，2-4 个词，便于以后按主题找。",
    )
    importance: int = Field(
        default=1,
        ge=1,
        le=5,
        description="重要程度 1-5。普通八卦用 1-2，真的要紧才用 4-5。",
    )


class OhmoSaveGossipTool(BaseTool):
    """Archive one piece of group gossip, with a gist, into the group's store."""

    name = "ohmo_save_gossip"
    description = (
        "把群里刚转来的可留档情报存进屎库。"
        "只有当这条消息里确实含有具体的瓜、人、事、时间、链接、截图或聊天记录转发时才调用。"
        "寒暄、闲聊、表情、没有信息量的吐槽、单纯的提问一律不要调用。"
        "原文由运行时提供，不要在参数里复述原文；"
        "转写里有多条时用 which 指认要存哪一条。"
    )
    input_model = OhmoSaveGossipInput

    def __init__(self, *, workspace: str | Path | None) -> None:
        self._workspace = workspace

    def is_read_only(self, arguments: OhmoSaveGossipInput) -> bool:
        # Permission is enforced by the channel guard in execute(): the tool only
        # acts when the runtime injected a QQ group request for this turn, and it
        # only ever writes into the gateway's own workspace store.
        #
        # Declaring it read-only is also the only way it can run at all. The
        # gateway builds its runtime without a permission prompt, so a tool that
        # reports itself as writing is failed before execute() is ever reached.
        del arguments
        return True

    async def execute(
        self,
        arguments: OhmoSaveGossipInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        request = context.metadata.get("ohmo_gossip_request")
        if not isinstance(request, dict):
            return ToolResult(output="ohmo_save_gossip 只能在 QQ 群消息里使用。", is_error=True)

        raw_text, error = _select_source(request, arguments.which)
        if error is not None:
            return ToolResult(output=error, is_error=True)

        media = [str(path) for path in (request.get("media") or [])]
        if not raw_text.strip() and not media:
            return ToolResult(output="这条消息没有可存档的内容。", is_error=True)

        try:
            # flock plus a directory scan are blocking, and every gateway session
            # shares one event loop -- without this the whole gateway stalls.
            result = await asyncio.to_thread(
                save_gossip_entry,
                self._workspace,
                channel=str(request.get("channel") or ""),
                chat_id=str(request.get("chat_id") or ""),
                raw_text=raw_text,
                gist=arguments.gist,
                title=arguments.title,
                tags=tuple(arguments.tags),
                importance=arguments.importance,
                source_message_ids=[str(value) for value in (request.get("message_ids") or [])],
                media_paths=media,
                sender_member_id=str(request.get("sender_member_id") or ""),
                sender_display_name=request.get("sender_display_name"),
                created_at=request.get("created_at"),
                raw_payload=request.get("raw_payload"),
            )
        except Exception as exc:
            logger.exception("gossip save failed")
            return ToolResult(output=f"存档失败：{exc}", is_error=True)

        # Deliberately not `media` / `paths`: _extract_tool_media reads those as
        # a request to push an extra channel message, which would spend another
        # passive-reply slot to say nothing.
        metadata: dict[str, Any] = {"gossip_path": str(result.path)}
        if result.duplicate:
            return ToolResult(output="这条屎已经有记录了。", metadata=metadata)
        if result.sensitive:
            return ToolResult(output="记下了（内容疑似含密钥，已标记）。", metadata=metadata)
        return ToolResult(output=f"记下了：{result.gist}", metadata=metadata)


def _select_source(request: Mapping[str, Any], which: int | None) -> tuple[str, str | None]:
    """Return the verbatim text to archive, or an error message.

    The model names a section by index; the text itself is read from the copy the
    runtime injected. That split is the whole design -- the model is allowed to
    summarise, but it never gets to hand us the original, so what lands in the
    archive is exactly what QQ sent.
    """
    if which is None:
        return str(request.get("raw_text") or ""), None

    segments = request.get("segments")
    if not isinstance(segments, list) or not segments:
        return "", "这条消息里没有可指认的编号（没有收到群聊转写）。"

    available: list[str] = []
    for segment in segments:
        if not isinstance(segment, Mapping):
            continue
        index = segment.get("index")
        if not isinstance(index, int):
            continue
        available.append(str(index))
        if index == which:
            return str(segment.get("content") or ""), None

    return "", f"没有第 {which} 条。可选的编号：{'、'.join(available) or '无'}。"


class OhmoListGossipInput(BaseModel):
    """Arguments selected by the model for listing the archive."""

    query: str | None = Field(
        default=None,
        description=(
            "想找什么，用自然语言或关键词写，例如“显卡降价”“上次那个截图”。"
            "不填就按时间从新到旧列出最近的条目。"
        ),
    )
    limit: int = Field(
        default=_DEFAULT_LIST_LIMIT,
        ge=1,
        le=_MAX_LIST_LIMIT,
        description=f"最多列几条，默认 {_DEFAULT_LIST_LIMIT}。",
    )


class OhmoListGossipTool(BaseTool):
    """List or search the current group's gossip archive, in the group itself."""

    name = "ohmo_list_gossip"
    description = (
        "列出或检索本群屎库里存了哪些东西。"
        "有人问“屎库里有什么”“存过什么”“有没有关于 X 的”时调用。"
        "列表由工具直接发到群里，你只要回一句极短的话（例如“上面就是。”），不要复述列表内容。"
        "只用当前群自己的存档，不会列其他群。"
    )
    input_model = OhmoListGossipInput

    def __init__(
        self,
        *,
        workspace: str | Path | None,
        publish_gossip_text: PublishGossipText | None = None,
    ) -> None:
        self._workspace = workspace
        self._publish_gossip_text = publish_gossip_text

    def is_read_only(self, arguments: OhmoListGossipInput) -> bool:
        # Read-only in the sense the permission layer means: no file is written
        # and no permission prompt is asked for. The group message it sends is
        # the whole point of the tool. Declaring otherwise would fail the call
        # before execute() -- the gateway builds its runtime with no prompt.
        del arguments
        return True

    async def execute(
        self,
        arguments: OhmoListGossipInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        request = _group_request(context)
        if request is None:
            return ToolResult(output="ohmo_list_gossip 只能在 QQ 群消息里使用。", is_error=True)
        if self._publish_gossip_text is None:
            return ToolResult(output="当前配置不能把屎库发到群里。", is_error=True)

        channel = str(request.get("channel") or "")
        chat_id = str(request.get("chat_id") or "")
        query = " ".join(str(arguments.query or "").split())

        try:
            # The directory scan and the body reads are blocking, and every
            # gateway session shares one event loop.
            headers = await asyncio.to_thread(
                _collect_entries,
                self._workspace,
                channel=channel,
                chat_id=chat_id,
                query=query,
                limit=arguments.limit,
            )
        except Exception as exc:
            logger.exception("gossip listing failed")
            return ToolResult(output=f"列屎库失败：{exc}", is_error=True)

        if not headers:
            return ToolResult(
                output=(
                    f"本群屎库里没有匹配“{query}”的条目。直接告诉群里没找到，不要编。"
                    if query
                    else "本群屎库还是空的。直接告诉群里还没存过东西。"
                )
            )

        scope = f"搜“{query}”" if query else "本群"
        listing = format_gossip_listing(headers, title=f"屎库 · {scope} 共 {len(headers)} 条")
        for message in format_gossip_messages(listing):
            await _publish(self._publish_gossip_text, chat_id, channel, message)

        # Which entry each number refers to, so "看第 2 条" resolves without the
        # model having to repeat anything back. The same mapping goes into the
        # tool result, which is what survives a restart -- tool_metadata is
        # dropped by the session snapshot's key whitelist, conversation history
        # is not.
        hits = {
            str(index): _entry_key(header)
            for index, header in enumerate(headers, start=1)
        }
        _remember_hits(context, hits)

        keys = " ".join(f"{index}={key}" for index, key in hits.items())
        return ToolResult(
            output=(
                f"已把本群屎库的 {len(headers)} 条发到群里了。不要再复述列表，"
                "只回一句极短的话（也可以什么都不说）。\n"
                f"序号→id：{keys}\n"
                "用户说“看第 N 条”时，用这里的 id 调 ohmo_show_gossip。"
            )
        )


class OhmoShowGossipInput(BaseModel):
    """Arguments selected by the model for showing one archived entry."""

    id: str | None = Field(
        default=None,
        description=(
            "要发的条目 id，取自上一次 ohmo_list_gossip 返回的“序号→id”。能用就用它。"
        ),
    )
    index: int | None = Field(
        default=None,
        ge=1,
        description="上一次列表里的序号（用户说“看第 N 条”）。拿不到 id 时才用它。",
    )


class OhmoShowGossipTool(BaseTool):
    """Send one archived entry back to the group, exactly as it was stored."""

    name = "ohmo_show_gossip"
    description = (
        "把屎库里某一条的原文一字不改地发回群里。"
        "用户说“看第 N 条”“把那条发出来”“发一下那个”时调用。"
        "参数用上一次 ohmo_list_gossip 返回的 id；没有 id 时才用 index。"
        "发完什么都不要再说：不要转述、不要复述、不要评论、不要加总结。"
    )
    input_model = OhmoShowGossipInput

    def __init__(
        self,
        *,
        workspace: str | Path | None,
        publish_gossip_text: PublishGossipText | None = None,
    ) -> None:
        self._workspace = workspace
        self._publish_gossip_text = publish_gossip_text

    def is_read_only(self, arguments: OhmoShowGossipInput) -> bool:
        del arguments
        return True

    async def execute(
        self,
        arguments: OhmoShowGossipInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        request = _group_request(context)
        if request is None:
            return ToolResult(output="ohmo_show_gossip 只能在 QQ 群消息里使用。", is_error=True)
        if self._publish_gossip_text is None:
            return ToolResult(output="当前配置不能把屎库发到群里。", is_error=True)

        channel = str(request.get("channel") or "")
        chat_id = str(request.get("chat_id") or "")

        entry_key = str(arguments.id or "").strip()
        if not entry_key and arguments.index is not None:
            entry_key = _lookup_index(context, arguments.index)
            if not entry_key:
                return ToolResult(
                    output=(
                        f"我这边没有第 {arguments.index} 条的记录（列过一次之后重启过就会这样）。"
                        "先用 ohmo_list_gossip 列一次，再用返回的 id 调我。"
                    ),
                    is_error=True,
                )
        if not entry_key:
            return ToolResult(
                output="需要一个 id：先用 ohmo_list_gossip 列出条目。",
                is_error=True,
            )

        try:
            found = await asyncio.to_thread(
                find_gossip_entry,
                self._workspace,
                channel=channel,
                chat_id=chat_id,
                entry_id=entry_key,
            )
            if found is None:
                return ToolResult(
                    output=f"本群屎库里没有 {entry_key} 这一条，先用 ohmo_list_gossip 确认一下。",
                    is_error=True,
                )
            entry = await asyncio.to_thread(load_gossip_entry, found)
        except Exception as exc:
            logger.exception("gossip load failed")
            return ToolResult(output=f"读这条失败：{exc}", is_error=True)

        messages = format_gossip_messages(format_gossip_body(entry))
        if not messages:
            return ToolResult(output="这一条没有可发送的正文。", is_error=True)
        for message in messages:
            await _publish(self._publish_gossip_text, chat_id, channel, message)

        return ToolResult(
            output=(
                f"已把 {entry.id or entry.path.name} 的原文原封不动发到群里了"
                f"（{len(messages)} 条消息）。不要再复述、不要再评论。"
            )
        )


def _group_request(context: ToolExecutionContext) -> Mapping[str, Any] | None:
    """Return this turn's QQ group context, or ``None`` outside a QQ group."""
    request = context.metadata.get("ohmo_gossip_request")
    return request if isinstance(request, Mapping) else None


def _collect_entries(
    workspace: str | Path | None,
    *,
    channel: str,
    chat_id: str,
    query: str,
    limit: int,
) -> list[MemoryHeader]:
    """Return the entries to list: search hits, or the newest ``limit``."""
    if query:
        return search_gossip(
            workspace,
            channel=channel,
            chat_id=chat_id,
            query=query,
            limit=limit,
        )
    return list_gossip_headers(workspace, channel=channel, chat_id=chat_id, max_files=limit)


def _entry_key(header: MemoryHeader) -> str:
    """Return the handle a listing gives the model for one entry.

    The id when the entry has one, otherwise its filename -- a hand-written
    entry has no id in its frontmatter and would otherwise be unreachable.
    """
    return str(header.id or header.path.name)


def _remember_hits(context: ToolExecutionContext, hits: dict[str, str]) -> None:
    """Record ``index -> entry key`` for the rest of this session.

    ``ToolExecutionContext.metadata`` is a shallow copy of the engine's tool
    metadata, so a top-level assignment here would be discarded; mutating the
    nested dict the runtime put there is what reaches the next turn.
    """
    stored = context.metadata.get("ohmo_gossip_hits")
    if isinstance(stored, dict):
        stored.clear()
        stored.update(hits)


def _lookup_index(context: ToolExecutionContext, index: int) -> str:
    """Return the entry key a previous listing numbered ``index``."""
    stored = context.metadata.get("ohmo_gossip_hits")
    if not isinstance(stored, Mapping):
        return ""
    return str(stored.get(str(index)) or "")


async def _publish(publish: PublishGossipText, chat_id: str, channel: str, content: str) -> None:
    """Send one already-chunked message; the publisher may be sync or async."""
    published = publish(chat_id, content, channel)
    if asyncio.iscoroutine(published):
        await published


__all__ = [
    "OhmoListGossipInput",
    "OhmoListGossipTool",
    "OhmoSaveGossipInput",
    "OhmoSaveGossipTool",
    "OhmoShowGossipInput",
    "OhmoShowGossipTool",
    "PublishGossipText",
]
