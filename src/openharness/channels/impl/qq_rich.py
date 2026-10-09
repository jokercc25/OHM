"""Flatten rich QQ group payloads into text plus downloadable attachments.

This module is deliberately free of any ``botpy`` import so it can be unit
tested without the SDK installed. :mod:`openharness.channels.impl.qq` feeds it
the raw event payload (``payload["d"]``) and handles all IO.

The QQ element shapes below follow the official group-message documentation but
are **not yet confirmed against a live payload**. :func:`census` exists to
settle that: run one @-mention per message kind and read the census line. Until
then, unknown element types degrade to an explicit marker rather than being
silently dropped.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openharness.utils.helpers import safe_filename

# QQ msg_elements[].message_type values we know how to render. Anything else is
# surfaced as a marker so the Phase-0 census can tell us what we are missing.
MSG_TYPE_TEXT = 0
MSG_TYPE_RICH = 1
MSG_TYPE_MARKDOWN = 2
MSG_TYPE_ARK = 3
MSG_TYPE_EMBED = 4
MSG_TYPE_MEDIA = 7
MSG_TYPE_CHAT_RECORD = 102
MSG_TYPE_QUOTE = 103

# Keys that carry short-lived credentials or server bookkeeping. QQ media URLs
# embed an rkey that expires, so they must never be persisted (download now).
_SENSITIVE_KEYS = frozenset({"auth_token", "authToken", "msg_idx", "ref_msg_idx", "token"})
_RKEY_RE = re.compile(r"(rkey=)[^&\s\"']+")
_UNSAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")

_REDACTED = "<redacted>"

_MAX_ELEMENT_DEPTH = 2
_MAX_ELEMENTS = 20

# Best-effort field priority for Ark / embed cards. 小黑盒 lands in tuwen/feed
# and carries title + desc + jump_url; jump_url is the highest-value field.
_CARD_FIELD_PRIORITY = (
    "title",
    "desc",
    "description",
    "jump_url",
    "url",
    "source",
    "tag",
    "nickname",
)


@dataclass(frozen=True)
class RichContent:
    """Result of flattening one QQ group payload."""

    text: str
    """Markdown transcript to append to the inbound message content."""

    media_urls: tuple[str, ...] = ()
    """Remote attachment URLs to download. This module never performs IO."""

    message_ids: tuple[str, ...] = ()
    """Every QQ message id seen (outer id plus each element's), for dedup."""

    attachments: tuple[dict[str, Any], ...] = ()
    """Normalised attachment metadata for logging and the sidecar."""

    has_rich: bool = False
    """True when cards/elements/attachments were flattened."""

    raw: dict[str, Any] | None = None
    """Redacted payload, for the optional sidecar file."""

    @property
    def is_empty(self) -> bool:
        return not self.text and not self.media_urls


def redact_payload(value: Any) -> Any:
    """Recursively strip short-lived credentials and bookkeeping keys."""
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            if name in _SENSITIVE_KEYS:
                redacted[name] = _REDACTED
            else:
                redacted[name] = redact_payload(item)
        return redacted
    if isinstance(value, str):
        return _RKEY_RE.sub(r"\1" + _REDACTED, value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_payload(item) for item in value]
    return value


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _text_of(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("content", "text", "desc", "title"):
            found = value.get(key)
            if isinstance(found, str) and found.strip():
                return found.strip()
    return ""


def _sender_label(element: Mapping[str, Any]) -> str:
    author = _as_mapping(element.get("author"))
    for key in ("username", "nickname", "member_openid", "id"):
        found = author.get(key)
        if isinstance(found, str) and found.strip():
            return found.strip()
    return "?"


def _render_card(element: Mapping[str, Any]) -> str:
    """Render an Ark / embed card down to title, summary and URL."""
    ark = _as_mapping(element.get("ark_data") or element.get("ark"))
    name = str(ark.get("ark_name") or ark.get("template_name") or "").strip()
    fields: dict[str, Any] = {}
    for container in (ark, element):
        for key, item in _as_mapping(container).items():
            fields.setdefault(str(key), item)
    kv = _as_mapping(ark.get("kv")) or _as_mapping(ark.get("fields"))
    for key, item in kv.items():
        fields.setdefault(str(key), item)

    lines = [f"[链接卡片{': ' + name if name else ''}]"]
    for key in _CARD_FIELD_PRIORITY:
        found = _text_of(fields.get(key))
        if found:
            lines.append(f"{key}: {found}")
    if len(lines) == 1:
        lines.append(f"[卡片字段未识别: {sorted(fields)[:8]}]")
    return "\n".join(lines)


def _render_chat_record(element: Mapping[str, Any], depth: int) -> str:
    """Render a forwarded chat record as a quoted transcript."""
    lines = ["[转发聊天记录]"]
    for item in _walk_elements(element, depth + 1):
        sender = _sender_label(item)
        body = _text_of(item) or _render_element(item, depth + 1)
        if body:
            lines.append(f"> {sender}: {body}")
    if len(lines) == 1:
        lines.append("> [记录内容未识别]")
    return "\n".join(lines)


def _render_quote(element: Mapping[str, Any], depth: int) -> str:
    reference = _as_mapping(element.get("message_reference") or element.get("reference"))
    quoted = _text_of(reference) or _text_of(element.get("reference_content"))
    if quoted:
        return f"> 引用: {quoted}"
    return "> [引用内容未识别]"


def _render_element(element: Any, depth: int = 0) -> str:
    """Render one msg_elements entry, recursing into forwarded records."""
    item = _as_mapping(element)
    if not item:
        text = _text_of(element)
        return text

    raw_type = item.get("message_type", item.get("type"))
    try:
        msg_type = int(raw_type)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        msg_type = -1

    if msg_type in (MSG_TYPE_TEXT, MSG_TYPE_MARKDOWN):
        return _text_of(item)
    if msg_type == MSG_TYPE_CHAT_RECORD:
        return _render_chat_record(item, depth) if depth < _MAX_ELEMENT_DEPTH else ""
    if msg_type == MSG_TYPE_QUOTE:
        return _render_quote(item, depth)
    if msg_type in (MSG_TYPE_ARK, MSG_TYPE_EMBED, MSG_TYPE_RICH):
        if item.get("ark_data") or item.get("ark"):
            return _render_card(item)
        return _text_of(item)
    if msg_type == MSG_TYPE_MEDIA:
        return "[媒体]"
    # Unknown type: keep an explicit marker so the Phase-0 census reveals it.
    return _text_of(item) or f"[未知元素 type={raw_type!r}]"


def _immediate_elements(container: Any) -> list[Mapping[str, Any]]:
    """Return the container's direct ``msg_elements``, non-recursively."""
    items = _as_mapping(container).get("msg_elements") or ()
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        return []
    return [mapped for item in list(items)[:_MAX_ELEMENTS] if (mapped := _as_mapping(item))]


def _walk_elements(element: Any, depth: int = 0) -> list[Mapping[str, Any]]:
    """Collect an element plus its nested ``msg_elements``, bounded in size."""
    if depth > _MAX_ELEMENT_DEPTH:
        return []
    collected: list[Mapping[str, Any]] = []
    for nested in _immediate_elements(element):
        if len(collected) >= _MAX_ELEMENTS:
            break
        collected.append(nested)
        collected.extend(_walk_elements(nested, depth + 1))
    return collected


def _collect_message_ids(raw: Mapping[str, Any]) -> tuple[str, ...]:
    ids: list[str] = []

    def add(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in ids:
            ids.append(text)

    add(raw.get("id"))
    for element in _walk_elements(raw):
        add(element.get("id") or element.get("msg_id"))
    return tuple(ids)


def _normalize_attachment(attachment: Any) -> dict[str, Any]:
    item = _as_mapping(attachment)
    return {
        "content_type": str(item.get("content_type") or ""),
        "filename": str(item.get("filename") or ""),
        "size": item.get("size"),
        "url": str(item.get("url") or ""),
    }


def is_downloadable_image(attachment: Mapping[str, Any]) -> bool:
    """Return whether *attachment* looks like an image we can fetch."""
    content_type = str(attachment.get("content_type") or "").lower()
    if content_type.startswith("image/"):
        return True
    name = str(attachment.get("filename") or "").lower()
    return name.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"))


def safe_attachment_name(msg_id: str, index: int, filename: str) -> str:
    """Build a deterministic, filesystem-safe attachment filename.

    The message id gets a flat character scrub rather than ``safe_filename``:
    an id is an opaque token, not a path, so ``Path(...).name`` would silently
    discard everything before a slash. ``filename`` *is* a filename, so there
    ``safe_filename``'s basename handling is exactly what strips a traversal
    attempt like ``../../etc/passwd``.
    """
    stem = _UNSAFE_STEM_RE.sub("_", str(msg_id or "")).strip("._-")[:48] or "msg"
    name = safe_filename(filename, max_length=64) or "image.jpg"
    return f"{stem}_{index}_{name}"


def _render_recent_messages(scene_ext: Any) -> str:
    """Render ``message_scene.ext`` (the '最近 N 条' context) as a transcript."""
    ext = _as_mapping(scene_ext)
    entries = ext.get("msg_elements") or ext.get("messages") or ()
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return ""
    lines = ["[群内最近消息]"]
    for item in list(entries)[:_MAX_ELEMENTS]:
        sender = _sender_label(_as_mapping(item))
        body = _render_element(item)
        if body:
            lines.append(f"> {sender}: {body}")
    return "\n".join(lines) if len(lines) > 1 else ""


def flatten_group_payload(raw: Mapping[str, Any] | None) -> RichContent:
    """Flatten a raw QQ group event payload into text, URLs and ids.

    Pure: performs no network or filesystem IO. Downloading ``media_urls`` and
    writing files is the caller's job.
    """
    if not isinstance(raw, Mapping):
        return RichContent(text="")

    blocks: list[str] = []
    urls: list[str] = []
    attachments: list[dict[str, Any]] = []

    scene = _as_mapping(raw.get("message_scene"))
    if recent := _render_recent_messages(scene.get("ext")):
        blocks.append(recent)

    elements = _immediate_elements(raw)
    for element in elements:
        rendered = _render_element(element)
        if rendered:
            blocks.append(f"{_sender_label(element)}: {rendered}")

    top_ark = raw.get("ark_data")
    if top_ark:
        blocks.append(_render_card({"ark_data": top_ark}))

    for attachment in raw.get("attachments") or ():
        meta = _normalize_attachment(attachment)
        if not meta["url"]:
            continue
        attachments.append(meta)
        label = meta["filename"] or meta["content_type"] or "attachment"
        if is_downloadable_image(meta):
            urls.append(meta["url"])
            blocks.append(f"[图片: {label}]")
        else:
            blocks.append(f"[附件: {label} ({meta['content_type']})]")

    text = "\n\n".join(block for block in blocks if block.strip())
    has_rich = bool(elements) or bool(top_ark) or bool(attachments)
    return RichContent(
        text=text,
        media_urls=tuple(dict.fromkeys(urls)),
        message_ids=_collect_message_ids(raw),
        attachments=tuple(attachments),
        has_rich=has_rich,
        raw=redact_payload(dict(raw)),
    )


def census(raw: Mapping[str, Any] | None) -> str:
    """Return a one-line field signature of a raw payload.

    This is the Phase-0 instrument: it answers "does QQ actually send
    msg_elements / ark_data / message_scene?" without dumping the whole event.
    """
    if not isinstance(raw, Mapping):
        return "payload=absent"

    scene = _as_mapping(raw.get("message_scene"))
    ext = scene.get("ext")
    elements = _immediate_elements(raw)
    first_type = elements[0].get("message_type") if elements else None
    author = _as_mapping(raw.get("author"))

    def count(value: Any) -> int:
        return len(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else 0

    return (
        f"msg_id={raw.get('id')!r} message_type={raw.get('message_type')!r} "
        f"n_msg_elements={len(elements)} first_elem_type={first_type!r} "
        f"has_ark={bool(raw.get('ark_data'))} n_attachments={count(raw.get('attachments'))} "
        f"has_scene_ext={bool(ext)} scene_ext_type={type(ext).__name__} "
        f"n_mentions={count(raw.get('mentions'))} content_len={len(str(raw.get('content') or ''))} "
        f"author_keys={sorted(author)} timestamp={raw.get('timestamp')!r}"
    )
