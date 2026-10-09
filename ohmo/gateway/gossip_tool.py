"""ohmo-only gossip archive tool.

The trigger is the tool description itself -- there is no keyword list and no
regex. The model reads the message and decides whether it is worth keeping.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

from ohmo.gossip import save_gossip_entry

logger = logging.getLogger(__name__)


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
