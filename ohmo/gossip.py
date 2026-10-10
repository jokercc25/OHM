"""Group gossip archive for ohmo.

Group members forward news, screenshots and chat-record cards -- the group calls
it 屎 -- and this module keeps them. Each entry pairs a one-line gist with the
original text, so the archive can be searched later without going back to QQ.

Layout::

    <workspace>/gossip/<channel>/<safe_chat_id>/
      INDEX.md
      .gossip.lock
      entries/
        2026-10-09-103856-1a2b3c4d.md
        2026-10-09-103856-1a2b3c4d.raw.json

Three structural choices worth stating, because each one is load-bearing:

* **Per group, not per user.** ``session_key_for_message`` gives every member of
  a group their own session, so hanging the archive off a session or off
  personal memory would hand each member a private archive nobody else can
  search. The entire point is that the group shares one.

* **Entries live one level below the index.** ``scan_memory_files`` skips only
  ``MEMORY.md`` by name, so any other markdown file sitting in a scanned
  directory is treated as an entry. The index therefore sits above the entries
  it indexes rather than beside them.

* **The memory schema is reused unchanged.** ``type: reference`` is a real
  :data:`~openharness.memory.schema.MEMORY_TYPES` value and ``category`` is
  free-form, so archiving needs no schema change -- and entries come out already
  legible to ``scan_memory_files``, which is exactly what phase 2's search will
  build on.

Phase 2 reads that archive back out: :func:`load_gossip_entry` returns the
stored original untouched, and :func:`search_gossip` finds entries by free text.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openharness.memory.scan import scan_memory_files
from openharness.memory.search import find_relevant_memories
from openharness.memory.schema import (
    SCHEMA_VERSION,
    coerce_bool,
    coerce_int,
    coerce_str_list,
    compute_memory_signature,
    format_datetime,
    parse_datetime,
    render_memory_file,
    split_memory_file,
    utc_now,
)
from openharness.memory.team import scan_for_secrets
from openharness.memory.types import MemoryHeader
from openharness.utils.file_lock import exclusive_file_lock
from openharness.utils.fs import atomic_write_text
from openharness.utils.helpers import split_message

from ohmo.workspace import get_gossip_dir

logger = logging.getLogger(__name__)

GOSSIP_TYPE = "reference"
GOSSIP_SCOPE = "team"
GOSSIP_CATEGORY = "gossip"
GOSSIP_SOURCE = "qq_group"

_ENTRIES_DIRNAME = "entries"
_INDEX_FILENAME = "INDEX.md"
_LOCK_FILENAME = ".gossip.lock"

# Bounds. An archive that grows without limit is a disk-space bug waiting to
# happen, and a single pathological forward can otherwise dwarf everything else.
_MAX_BODY_CHARS = 20_000
_MAX_GIST_CHARS = 60
_GIST_TRUNCATE_TO = 40
_MAX_TITLE_CHARS = 60
_SCAN_LIMIT = 500

# Display bounds. QQ answers an inbound @-message with at most five passive
# replies, and a turn already spends one on the "thinking" progress line before
# any tool runs plus whatever the model says afterwards -- so the verbatim dump
# gets the remainder. Going over the budget does not queue: the extra messages
# are rejected outright, which loses the tail silently. A stated truncation is
# the better failure.
MAX_DISPLAY_MESSAGES = 3
DISPLAY_CHUNK_CHARS = 1_800
_LISTING_TITLE_CHARS = 40

_UNSAFE_PATH_RE = re.compile(r"[^A-Za-z0-9_.-]+")

# Transcript furniture, skipped when falling back to the original for a gist.
_SECTION_LINE_RE = re.compile(r"^=+\s*消息\s*\d+\s*=+$")
_MESSAGE_MARKER = "[消息内容]"

# Section headings ``compose_gossip_body`` writes. Reading them back is how the
# original is recovered without a parser for anything else.
_RAW_SECTION = "## 原文"
_MEDIA_SECTION = "## 图片"
_MEDIA_NOTE_SUFFIX = "（待视觉模型补描述）"
_NO_TEXT_PLACEHOLDER = "（无文本）"


@dataclass(frozen=True)
class GossipSaveResult:
    """Outcome of one archive write."""

    path: Path
    gist: str
    duplicate: bool
    sensitive: bool


@dataclass(frozen=True)
class GossipEntry:
    """One archived entry, read back off disk."""

    path: Path
    metadata: dict[str, Any]
    raw_text: str
    media_paths: tuple[str, ...]

    @property
    def id(self) -> str:
        """Return the entry's archive id, empty for a hand-written file."""
        return str(self.metadata.get("id") or "")

    @property
    def gist(self) -> str:
        return str(self.metadata.get("description") or self.metadata.get("gist") or "")

    @property
    def created_at(self) -> str:
        return str(self.metadata.get("created_at") or "")


def safe_segment(raw: str, *, default: str = "unknown") -> str:
    """Return a filesystem-safe path segment for an external identifier.

    ``chat_id`` and ``channel`` arrive from QQ, so neither may reach the
    filesystem unfiltered -- a value containing ``../`` would otherwise escape
    the gossip root.
    """
    return _UNSAFE_PATH_RE.sub("_", str(raw or "")).strip("._") or default


def gossip_group_dir(workspace: str | Path | None, *, channel: str, chat_id: str) -> Path:
    """Return the archive directory for one group."""
    return get_gossip_dir(workspace) / safe_segment(channel) / safe_segment(chat_id)


def gossip_entries_dir(workspace: str | Path | None, *, channel: str, chat_id: str) -> Path:
    """Return the entries directory for one group."""
    return gossip_group_dir(workspace, channel=channel, chat_id=chat_id) / _ENTRIES_DIRNAME


def gossip_index_path(workspace: str | Path | None, *, channel: str, chat_id: str) -> Path:
    """Return the human-readable index path for one group."""
    return gossip_group_dir(workspace, channel=channel, chat_id=chat_id) / _INDEX_FILENAME


def normalize_gist(raw_gist: str | None, raw_text: str, sender: str | None = None) -> str:
    """Return a clean one-line gist for an entry.

    The model supplies the gist, but it must never be the only path to a usable
    entry: a blank or nonsense value degrades to the first line of the original
    rather than failing the save. Losing the archive is much worse than losing
    the summary.
    """
    candidate = " ".join(str(raw_gist or "").split()).strip("\"'“”「」『』").strip()
    if not candidate:
        candidate = _fallback_gist(raw_text, sender)
    if len(candidate) > _MAX_GIST_CHARS:
        candidate = candidate[:_GIST_TRUNCATE_TO].rstrip() + "…"
    return candidate


def _fallback_gist(raw_text: str, sender: str | None) -> str:
    """Return a usable gist when the model supplied none.

    Transcript furniture has to be skipped here. A section header like
    ``=== 消息 2 ===`` makes a useless summary, and that is precisely the shape
    the fallback sees when a whole transcript is archived rather than one
    segment of it.
    """
    for line in raw_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped == "---":
            continue
        if _SECTION_LINE_RE.match(stripped):
            continue
        if stripped.startswith(_MESSAGE_MARKER):
            stripped = stripped[len(_MESSAGE_MARKER):].strip()
        if stripped:
            return " ".join(stripped.split())
    return f"{sender or '有人'}转发的内容"


def compose_gossip_body(raw_text: str, media_paths: Sequence[str] = ()) -> str:
    """Render the archived original, verbatim, plus image placeholders.

    The text is stored exactly as the channel captured it. The model contributes
    only the gist -- a summary is allowed to be lossy, the original is not, and
    that asymmetry is the whole reason the text is injected by the runtime
    instead of being passed in as a tool argument.
    """
    lines = [_RAW_SECTION, "", raw_text.strip() or _NO_TEXT_PLACEHOLDER, ""]
    if media_paths:
        lines.extend([_MEDIA_SECTION, ""])
        lines.extend(f"- {path}{_MEDIA_NOTE_SUFFIX}" for path in media_paths)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_gossip_entry(
    workspace: str | Path | None,
    *,
    channel: str,
    chat_id: str,
    raw_text: str,
    gist: str | None = None,
    title: str | None = None,
    tags: Sequence[str] = (),
    importance: int = 1,
    source_message_ids: Sequence[str] = (),
    media_paths: Sequence[str] = (),
    sender_member_id: str = "",
    sender_display_name: str | None = None,
    created_at: str | None = None,
    raw_payload: Mapping[str, Any] | None = None,
) -> GossipSaveResult:
    """Archive one piece of gossip and return where it landed.

    Synchronous and lock-guarded: callers on an event loop must defer it with
    ``asyncio.to_thread``, since the lock and the directory scan are both
    blocking and the gateway shares one loop across every session.
    """
    entries_dir = gossip_entries_dir(workspace, channel=channel, chat_id=chat_id)
    entries_dir.mkdir(parents=True, exist_ok=True)

    text = str(raw_text or "").strip()[:_MAX_BODY_CHARS]
    media = [str(path) for path in media_paths if str(path).strip()]
    message_ids = [str(value) for value in source_message_ids if str(value).strip()]
    clean_gist = normalize_gist(gist, text, sender_display_name)
    signature = compute_memory_signature(text, GOSSIP_TYPE, GOSSIP_CATEGORY)

    with exclusive_file_lock(entries_dir.parent / _LOCK_FILENAME):
        existing = _scan_entries(entries_dir, include_disabled=True, max_files=None)
        duplicate = _find_duplicate(existing, message_ids, signature)
        if duplicate is not None:
            return _merge_into_duplicate(duplicate, message_ids, importance)

        now = parse_datetime(created_at) or utc_now()
        stamp = now.strftime("%Y-%m-%d-%H%M%S")
        token = secrets.token_hex(4)
        path = entries_dir / f"{stamp}-{token}.md"
        sensitive = bool(scan_for_secrets(text))

        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "id": f"mem-{stamp.replace('-', '')}-{token}",
            "name": _entry_title(title, clean_gist),
            "description": clean_gist,
            "type": GOSSIP_TYPE,
            "scope": GOSSIP_SCOPE,
            "category": GOSSIP_CATEGORY,
            "importance": max(1, min(coerce_int(importance, default=1), 5)),
            "source": GOSSIP_SOURCE,
            # Computed over the original text rather than the rendered body, so
            # it stays stable if the body layout ever changes.
            "signature": signature,
            "created_at": format_datetime(now),
            "updated_at": format_datetime(now),
            "ttl_days": None,
            "disabled": False,
            "supersedes": [],
            "tags": list(coerce_str_list(tags)),
            "gist": clean_gist,
            "channel": str(channel),
            "chat_id": str(chat_id),
            "sender_member_id": str(sender_member_id or ""),
            "sender_display_name": str(sender_display_name or ""),
            "msg_ids": message_ids,
            "media": media,
            "sensitive": sensitive,
        }
        atomic_write_text(path, render_memory_file(metadata, compose_gossip_body(text, media)))

        if raw_payload:
            # The sidecar holds the redacted payload from the channel. QQ media
            # URLs carry a short-lived rkey and are never persisted.
            sidecar = entries_dir / f"{stamp}-{token}.raw.json"
            atomic_write_text(sidecar, json.dumps(dict(raw_payload), ensure_ascii=False, indent=2) + "\n")

    if sensitive:
        logger.warning(
            "gossip entry flagged sensitive channel=%s chat_id=%s path=%s",
            channel,
            chat_id,
            path,
        )
    build_gossip_index(workspace, channel=channel, chat_id=chat_id)
    logger.info(
        "gossip entry saved channel=%s chat_id=%s path=%s duplicate=false",
        channel,
        chat_id,
        path,
    )
    return GossipSaveResult(path=path, gist=clean_gist, duplicate=False, sensitive=sensitive)


def list_gossip_headers(
    workspace: str | Path | None,
    *,
    channel: str,
    chat_id: str,
    include_disabled: bool = False,
    max_files: int | None = _SCAN_LIMIT,
) -> list[MemoryHeader]:
    """List the entries of one group, newest first."""
    return _scan_entries(
        gossip_entries_dir(workspace, channel=channel, chat_id=chat_id),
        include_disabled=include_disabled,
        max_files=max_files,
    )


def find_gossip_entry(
    workspace: str | Path | None,
    *,
    channel: str,
    chat_id: str,
    entry_id: str,
) -> Path | None:
    """Return the entry ``entry_id`` names inside one group's archive.

    The lookup is a string comparison against this group's own scan -- never a
    path join. ``entry_id`` reaches here from the model, and an argument that
    resolved as a filesystem path would let it ask for any file on the box and
    have the result published into the group.

    The filename is accepted as well as the id so a hand-written entry, which
    has no ``id`` in its frontmatter, is still reachable.
    """
    wanted = str(entry_id or "").strip()
    if not wanted:
        return None
    for header in list_gossip_headers(
        workspace,
        channel=channel,
        chat_id=chat_id,
        include_disabled=True,
        max_files=None,
    ):
        if (header.id and header.id == wanted) or header.path.name == wanted:
            return header.path
    return None


def search_gossip(
    workspace: str | Path | None,
    *,
    channel: str,
    chat_id: str,
    query: str,
    limit: int = _SCAN_LIMIT,
) -> list[MemoryHeader]:
    """Return this group's entries overlapping ``query``, best match first.

    Delegates to :func:`~openharness.memory.search.find_relevant_memories` so
    the ranking (metadata hits weighted over body hits, plus importance and
    recency) and the query tokenizer are the same ones the personal memory store
    already uses -- including the per-Han-character split, which is what makes a
    Chinese query work at all without a segmenter.
    """
    entries_dir = gossip_entries_dir(workspace, channel=channel, chat_id=chat_id)
    if not entries_dir.is_dir():
        return []
    return find_relevant_memories(
        query,
        entries_dir,
        max_results=limit,
        memory_dir=entries_dir,
        max_files=_SCAN_LIMIT,
    )


def list_gossip_groups(workspace: str | Path | None) -> list[tuple[str, str]]:
    """Return ``(channel, chat_id)`` for every group with an archive on disk."""
    gossip_root = get_gossip_dir(workspace)
    groups: list[tuple[str, str]] = []
    if not gossip_root.is_dir():
        return groups
    for channel_dir in sorted(path for path in gossip_root.iterdir() if path.is_dir()):
        for group_dir in sorted(path for path in channel_dir.iterdir() if path.is_dir()):
            groups.append((channel_dir.name, group_dir.name))
    return groups


def load_gossip_entry(path: str | Path) -> GossipEntry:
    """Read one archived entry back, keeping its original text verbatim.

    Read-only. The text is sliced out of the stored body without being
    re-wrapped, re-indented or otherwise normalised: phase 1 wrote exactly what
    the channel captured, and reproducing anything else would defeat the point
    of having an archive.
    """
    entry_path = Path(path)
    metadata, body, _, _ = split_memory_file(entry_path.read_text(encoding="utf-8"))
    sections = _split_body_sections(body)
    return GossipEntry(
        path=entry_path,
        metadata=metadata,
        raw_text=sections.get(_RAW_SECTION, "").strip(),
        media_paths=_parse_media_section(sections.get(_MEDIA_SECTION, "")),
    )


def format_gossip_listing(headers: Sequence[MemoryHeader], *, title: str) -> str:
    """Render a numbered index of entries, newest first.

    Only metadata is used, so listing costs no body reads. The numbering is what
    the user refers back to ("看第 2 条"), which is why the display order and the
    order of the ids handed to the model must come from one sequence.
    """
    lines = [title]
    lines.extend(
        f"{index}. {_listing_line(header)}" for index, header in enumerate(headers, start=1)
    )
    return "\n".join(lines)


def format_gossip_body(entry: GossipEntry) -> str:
    """Return the exact text to publish for one entry.

    Images become placeholders rather than uploads: the archive stores local
    paths, and QQ's group API takes only a public URL, so the group is told what
    was attached instead of being shown it.
    """
    text = entry.raw_text
    if not entry.media_paths:
        return text
    placeholders = "\n".join(
        f"[图片{index}] {Path(path).name}"
        for index, path in enumerate(entry.media_paths, start=1)
    )
    return f"{text}\n\n{placeholders}" if text else placeholders


def format_gossip_messages(text: str) -> list[str]:
    """Split entry text into the messages that will carry it.

    Capped at :data:`MAX_DISPLAY_MESSAGES`, and the last one says so when the
    cap bites. A silently short dump would read as the whole record.
    """
    chunks = split_message(text, DISPLAY_CHUNK_CHARS)
    if len(chunks) <= MAX_DISPLAY_MESSAGES:
        return chunks
    kept = chunks[:MAX_DISPLAY_MESSAGES]
    sent = sum(len(chunk) for chunk in kept)
    kept[-1] = f"{kept[-1]}\n…（原文过长，已截断：共 {len(text)} 字，已发 {sent} 字）"
    return kept


def build_gossip_index(
    workspace: str | Path | None,
    *,
    channel: str,
    chat_id: str,
    headers: Sequence[MemoryHeader] | None = None,
) -> Path | None:
    """Rewrite ``INDEX.md`` from the entries directory.

    Rebuilt rather than appended to: the save path already scans the entries for
    dedup, so a full rewrite is nearly free and cannot drift away from what is
    actually on disk after a hand edit or a hand deletion.
    """
    group_dir = gossip_group_dir(workspace, channel=channel, chat_id=chat_id)
    if not group_dir.is_dir():
        return None
    if headers is None:
        headers = _scan_entries(group_dir / _ENTRIES_DIRNAME, include_disabled=True, max_files=None)

    ordered = sorted(headers, key=lambda header: (header.created_at, header.path.name))
    lines = [
        f"# 屎库 · {safe_segment(channel)}/{safe_segment(chat_id)}",
        "",
        f"- 条数：{len(ordered)}",
        "",
    ]
    for header in ordered:
        tags = f" [{'、'.join(header.tags)}]" if header.tags else ""
        state = "（已停用）" if header.disabled else ""
        date = header.created_at[:10] or "----------"
        link = f"{_ENTRIES_DIRNAME}/{header.path.name}"
        lines.append(f"- {date} [{header.title}]({link}){tags} — {header.description}{state}")

    index_path = group_dir / _INDEX_FILENAME
    atomic_write_text(index_path, "\n".join(lines).rstrip() + "\n")
    return index_path


def load_gossip_prompt(workspace: str | Path | None = None) -> str | None:
    """Return the prompt section describing the gossip archive.

    Renders only the directory and the counts -- never a gist, never a body.
    Gists are the model's own summaries of untrusted third-party content, so
    putting them in the system prompt would open an indirect prompt-injection
    path; and an index that grows without bound does not belong in a prompt
    anyway. The model can read entries on demand.
    """
    gossip_root = get_gossip_dir(workspace)
    groups = 0
    entries = 0
    if gossip_root.is_dir():
        for channel_dir in sorted(path for path in gossip_root.iterdir() if path.is_dir()):
            for group_dir in sorted(path for path in channel_dir.iterdir() if path.is_dir()):
                groups += 1
                entries_dir = group_dir / _ENTRIES_DIRNAME
                if entries_dir.is_dir():
                    entries += sum(1 for _ in entries_dir.glob("*.md"))

    return "\n".join(
        [
            "# 屎库(gossip archive)",
            f"- 目录：{gossip_root}（按群分目录，共 {groups} 个群、{entries} 条）",
            "- 群里有人转来情报时，先自己判断值不值得留：",
            "  值得留 = 具体的瓜、人、事、时间、链接、截图、聊天记录转发。",
            "  不值得留 = 寒暄、闲聊、表情、无信息量的吐槽、单纯的提问。",
            "- 值得留就调用 `ohmo_save_gossip` 存一条。`gist` 用一句中文（≤40 字）写清“谁、什么事”，"
            "这是别人以后检索的入口。",
            "- 群聊转写里有多条时，必须用 `which` 指明存哪一条：那几条通常是 @ 你之前群里各说各的，"
            "整段存会把不相干的也存进去。`gist`/`title`/`tags` 也都只描述你选中的那一条。",
            "- 不值得留就正常聊天，不要调用工具，也不要提屎库。",
            "- 存完只回一句极短的话（例如“记下了。”），不要复述内容。",
            "- 有人问屎库里有什么、或者想找某件事时，调用 `ohmo_list_gossip`："
            "它自己会把列表发到群里，你只要回一句极短的话（例如“上面就是。”），不要复述列表。",
            "- 列表每条前面的序号和 id：用户说“看第 N 条”就调 `ohmo_show_gossip(index=N)`，"
            "也可以把上一条列出来的 id 直接传给 `ohmo_show_gossip(id=...)`。id 更准，能用就用 id。",
            "- `ohmo_show_gossip` 会把存档原文一字不改地发到群里。发完什么都不要再说："
            "不要转述、不要复述、不要评论、不要加总结。",
        ]
    )


def _scan_entries(
    entries_dir: Path,
    *,
    include_disabled: bool = False,
    max_files: int | None = _SCAN_LIMIT,
) -> list[MemoryHeader]:
    """Scan one group's entries.

    ``scan_memory_files``'s first argument is only a fallback for when
    ``memory_dir`` is omitted; passing the entries directory explicitly is also
    what keeps this scan away from the rest of the workspace.
    """
    if not entries_dir.is_dir():
        return []
    return scan_memory_files(
        entries_dir,
        max_files=max_files,
        include_disabled=include_disabled,
        memory_dir=entries_dir,
    )


def _split_body_sections(body: str) -> dict[str, str]:
    """Split a stored body into its ``## `` sections, headings included.

    A forwarded chat record can itself contain a line starting with ``## ``,
    which would open a section that is not one. The damage is bounded: the text
    is only ever put back together for display, and ``## 原文`` is always the
    first heading ``compose_gossip_body`` writes.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in body.splitlines():
        if line.startswith("## "):
            current = line.strip()
            sections.setdefault(current, [])
            continue
        if current is not None:
            sections[current].append(line)
    return {name: "\n".join(lines) for name, lines in sections.items()}


def _parse_media_section(text: str) -> tuple[str, ...]:
    """Return the attachment paths listed under ``## 图片``."""
    paths: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("- "):
            continue
        value = stripped[2:].strip()
        if value.endswith(_MEDIA_NOTE_SUFFIX):
            value = value[: -len(_MEDIA_NOTE_SUFFIX)].strip()
        if value:
            paths.append(value)
    return tuple(paths)


def _listing_line(header: MemoryHeader) -> str:
    """Return one listing row: date, title, sender, attachment count.

    The date is the stored one, sliced out of the UTC timestamp. Rendering a
    clock time here would mean picking a timezone to lie about -- the container
    runs UTC while the group does not.
    """
    date = header.created_at[:10] or header.updated_at[:10] or "----------"
    parts = [date]
    title = " ".join(str(header.title or header.description or "").split())
    if title:
        parts.append(title[:_LISTING_TITLE_CHARS])
    sender = str(header.frontmatter.get("sender_display_name") or "").strip()
    if sender:
        parts.append(sender)
    count = len(coerce_str_list(header.frontmatter.get("media")))
    if count:
        parts.append(f"{count}图")
    return " ｜ ".join(parts)


def _entry_title(title: str | None, gist: str) -> str:
    """Return the short title an entry is indexed under.

    ``name`` becomes ``MemoryHeader.title``, which is what phase 2 will rank on,
    so it is normalised the same way the gist is instead of being taken as raw
    model free-text.
    """
    candidate = " ".join(str(title or "").split())
    if not candidate:
        candidate = gist
    return candidate[:_MAX_TITLE_CHARS].strip() or "未命名"


def _find_duplicate(
    headers: Sequence[MemoryHeader],
    message_ids: Sequence[str],
    signature: str,
) -> MemoryHeader | None:
    """Return the entry a new save duplicates, if there is one.

    Message ids are checked first because they are the only exact key: a
    re-forwarded image downloads under a fresh filename, so media paths identify
    nothing. The content signature is the fallback for a card that arrives as
    plain text the second time, with no ids to match on.
    """
    incoming = set(message_ids)
    if incoming:
        for header in headers:
            if incoming & set(coerce_str_list(header.frontmatter.get("msg_ids"))):
                return header
    if signature:
        for header in headers:
            if header.signature and header.signature == signature:
                return header
    return None


def _merge_into_duplicate(
    header: MemoryHeader,
    message_ids: Sequence[str],
    importance: int,
) -> GossipSaveResult:
    """Fold a re-forward into the entry that already covers it.

    The body is deliberately left alone. The first capture is the original; the
    second is the same event arriving twice, so only the bookkeeping moves. New
    media is not merged either -- the original already holds those images, and
    appending would just accumulate copies of the same file.
    """
    metadata, body, _, _ = split_memory_file(header.path.read_text(encoding="utf-8"))
    metadata["msg_ids"] = list(
        dict.fromkeys([*coerce_str_list(metadata.get("msg_ids")), *message_ids])
    )
    metadata["importance"] = max(coerce_int(metadata.get("importance"), default=0), importance)
    metadata["updated_at"] = format_datetime(utc_now())
    atomic_write_text(header.path, render_memory_file(metadata, body))
    logger.info("gossip entry duplicated path=%s merged_ids=%s", header.path, len(message_ids))
    return GossipSaveResult(
        path=header.path,
        gist=str(metadata.get("description") or ""),
        duplicate=True,
        sensitive=coerce_bool(metadata.get("sensitive"), default=False),
    )
