"""Flatten QQ group payloads into text plus downloadable attachments.

This module is deliberately free of any ``botpy`` import so it can be unit
tested without the SDK installed. :mod:`openharness.channels.impl.qq` feeds it
the raw event payload (``payload["d"]``) and handles all IO.

MEASURED SHAPE
--------------
Verified against live payloads on 2026-10-09 (qq-botpy 1.2.1 plus the
raw-payload hook in ``qq.py``). Every point below contradicts what the QQ
documentation implies, which is exactly why the flattening was held back until
a real capture existed:

1. ``msg_elements`` is a list holding **exactly one** element. It has no
   ``message_type`` and no ``author``. Its ``content`` is a plain-text
   transcript **QQ already produced for us**::

       === 消息 1 ===
       [消息内容] 唐完了

       === 消息 2 ===
       [消息内容] [群聊的聊天记录]
       [消息类型] 合并转发消息
       [关联消息]
       --- 第1条 ---
           [消息内容] 什么时候来着
           [发送者] 旺旺雪饼

   Merged-forward records therefore arrive pre-flattened with sender labels
   included, and image attachments arrive as inline text lines. There is no
   ``ark_data`` to walk and no element tree to recurse.

   This matters because the official bot only receives a group message when it
   is @-mentioned, so a forwarded card is never delivered as its own event --
   the transcript is the *only* channel by which it reaches us.

2. ``message_scene.ext`` is a list of three **credential** strings
   (``msg_idx=…``, ``auth_token=…``, ``ref_msg_idx=…``) used to reply with a
   quote. It is *not* the recent-message list, and carries nothing archivable.

3. Image URLs embed a short-lived ``rkey``. They must be fetched immediately
   and never persisted; the dump and the archive sidecar both carry the
   redacted form.

4. Transcripts are not guaranteed. A bare @-mention can arrive with
   ``msg_elements: null``, so every consumer must tolerate an empty result.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from openharness.utils.helpers import safe_filename

# ``=== 消息 N ===`` section header emitted once per message in a transcript.
_SECTION_RE = re.compile(r"^===\s*消息\s*(\d+)\s*===\s*$", re.MULTILINE)

# Inline attachment line, e.g.
#   [附件1] 类型:图片 文件名:a.jpg 尺寸:261x264 大小:8.8KB URL:https://…&rkey=…
# QQ serves images from /download with no file extension, so the 类型 marker --
# not the URL -- is the only reliable signal that a line is an image.
_ATTACHMENT_LINE_RE = re.compile(r"\[附件\d+\]")
_FILENAME_RE = re.compile(r"文件名[:：]\s*(\S+)")
_URL_RE = re.compile(r"URL[:：]\s*(\S+)")

# Keys carrying short-lived credentials or server bookkeeping.
_SENSITIVE_KEYS = frozenset({"auth_token", "authToken", "msg_idx", "ref_msg_idx", "token"})
_RKEY_RE = re.compile(r"(rkey=)[^&\s\"']+")
_UNSAFE_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")

_REDACTED = "<redacted>"

# Bounds, so a pathological payload cannot blow up the prompt.
_MAX_TRANSCRIPT_CHARS = 8000
_MAX_SEGMENTS = 20


@dataclass(frozen=True)
class MediaRef:
    """A downloadable image referenced inline by a transcript."""

    url: str
    filename: str = ""


@dataclass(frozen=True)
class TranscriptSegment:
    """One ``=== 消息 N ===`` section of a QQ transcript."""

    index: int
    """The ``N`` from the section header, as QQ numbered it."""

    content: str
    """The section body, verbatim. This is the archival original: it must reach
    the archive untouched, never through a model paraphrase."""


@dataclass(frozen=True)
class RichContent:
    """Result of flattening one QQ group payload."""

    text: str
    """QQ's transcript, to be appended to the inbound message content."""

    segments: tuple[TranscriptSegment, ...] = ()
    """The transcript split by section, so a caller can name one verbatim."""

    media: tuple[MediaRef, ...] = ()
    """Inline images found in the transcript. This module never performs IO."""

    message_ids: tuple[str, ...] = ()
    """QQ message ids seen, for dedup across re-forwards."""

    has_rich: bool = False
    """True when a transcript was present."""

    raw: dict[str, Any] | None = None
    """Redacted payload, for the optional dump and archive sidecar."""

    @property
    def media_urls(self) -> tuple[str, ...]:
        return tuple(ref.url for ref in self.media)

    @property
    def is_empty(self) -> bool:
        return not self.text and not self.media


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
    if isinstance(value, (list, tuple)):
        return [redact_payload(item) for item in value]
    return value


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _transcript_of(raw: Mapping[str, Any]) -> str:
    """Return QQ's plain-text transcript for this event, or ``""``."""
    elements = raw.get("msg_elements")
    if not isinstance(elements, list):
        return ""
    for element in elements:
        content = _as_mapping(element).get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()[:_MAX_TRANSCRIPT_CHARS]
    return ""


def split_transcript(text: str) -> tuple[TranscriptSegment, ...]:
    """Split a QQ transcript into its ``=== 消息 N ===`` sections.

    Bodies are returned verbatim so the archive can quote them word for word.
    A transcript with no section markers yields no segments rather than one
    giant one -- callers treat "no segments" as "nothing was quoted".
    """
    if not text:
        return ()
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        return ()

    segments: list[TranscriptSegment] = []
    for position, match in enumerate(matches):
        if len(segments) >= _MAX_SEGMENTS:
            break
        start = match.end()
        end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        segments.append(
            TranscriptSegment(index=_as_int(match.group(1), position + 1), content=body)
        )
    return tuple(segments)


def extract_media_refs(transcript: str) -> tuple[MediaRef, ...]:
    """Pull downloadable image references out of a transcript.

    Only lines carrying the ``类型:图片`` marker are collected: QQ's image URLs
    end at ``/download`` with no extension, so the URL alone cannot tell an
    image from a voice clip.
    """
    refs: list[MediaRef] = []
    seen: set[str] = set()
    for line in transcript.splitlines():
        if not _ATTACHMENT_LINE_RE.search(line) or "图片" not in line:
            continue
        url_match = _URL_RE.search(line)
        if not url_match:
            continue
        url = url_match.group(1)
        if url in seen:
            continue
        seen.add(url)
        name_match = _FILENAME_RE.search(line)
        refs.append(MediaRef(url=url, filename=name_match.group(1) if name_match else ""))
    return tuple(refs)


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


def flatten_group_payload(raw: Mapping[str, Any] | None) -> RichContent:
    """Flatten a raw QQ group event payload into text, segments and media.

    Pure: performs no network or filesystem IO. Downloading ``media_urls`` and
    writing files is the caller's job. ``media_urls`` are the *live* URLs -- the
    ``rkey`` they carry expires within minutes, which is why the redacted copy
    in ``raw`` is kept separate from the fetchable one.
    """
    if not isinstance(raw, Mapping):
        return RichContent(text="")

    transcript = _transcript_of(raw)
    outer_id = str(raw.get("id") or "").strip()
    return RichContent(
        text=transcript,
        segments=split_transcript(transcript),
        media=extract_media_refs(transcript),
        message_ids=(outer_id,) if outer_id else (),
        has_rich=bool(transcript),
        raw=redact_payload(dict(raw)),
    )


def rewrite_media_links(transcript: str, saved: Mapping[str, str]) -> str:
    """Swap every image line's live URL for the local file it was saved to.

    The ``rkey`` in a QQ media URL stops working within minutes, so archiving
    the URL verbatim would file away a link that is dead on arrival. ``saved``
    maps URL -> local path for the downloads that landed; anything missing is
    marked as failed rather than left pointing at the expired CDN.
    """
    if not transcript:
        return transcript

    def replace(match: re.Match[str]) -> str:
        url = match.group(1)
        path = saved.get(url)
        return f"本地文件：{path}" if path else "[图片下载失败]"

    # keepends: this text is archived verbatim, so line endings must survive.
    lines = []
    for line in transcript.splitlines(keepends=True):
        if _ATTACHMENT_LINE_RE.search(line) and "图片" in line and _URL_RE.search(line):
            line = _URL_RE.sub(replace, line, count=1)
        lines.append(line)
    return "".join(lines)


def census(raw: Mapping[str, Any] | None) -> str:
    """Return a one-line field signature of a raw payload.

    The Phase-0 instrument: it answered "does QQ actually send a transcript?"
    without dumping whole events. Kept because it is the cheapest way to spot
    the shape changing under us.
    """
    if not isinstance(raw, Mapping):
        return "payload=absent"

    transcript = _transcript_of(raw)
    scene_ext = _as_mapping(raw.get("message_scene")).get("ext")
    elements = raw.get("msg_elements")
    author = _as_mapping(raw.get("author"))

    def count(value: Any) -> int:
        return len(value) if isinstance(value, (list, tuple)) else 0

    return (
        f"msg_id={raw.get('id')!r} message_type={raw.get('message_type')!r} "
        f"n_msg_elements={count(elements)} transcript_len={len(transcript)} "
        f"n_segments={len(split_transcript(transcript))} "
        f"n_images={len(extract_media_refs(transcript))} "
        f"has_ark={bool(raw.get('ark_data'))} n_attachments={count(raw.get('attachments'))} "
        f"n_scene_ext={count(scene_ext)} n_mentions={count(raw.get('mentions'))} "
        f"content_len={len(str(raw.get('content') or ''))} "
        f"author_keys={sorted(author)} timestamp={raw.get('timestamp')!r}"
    )
