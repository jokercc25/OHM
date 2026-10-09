"""Tests for QQ rich-payload flattening.

This file must not import ``botpy``: ``qq_rich`` is deliberately SDK-free so the
flattening logic (the part most likely to be wrong) stays testable everywhere.

The fixtures below are trimmed copies of payloads captured live on 2026-10-09,
not reconstructions from the QQ docs. That distinction is the whole point of the
file: the documented shape and the real shape disagreed on every field that
mattered, so these tests pin the measured behaviour.
"""

from __future__ import annotations

from openharness.channels.impl.qq_rich import (
    MediaRef,
    RichContent,
    census,
    extract_media_refs,
    flatten_group_payload,
    redact_payload,
    rewrite_media_links,
    safe_attachment_name,
    split_transcript,
)

# A transcript with the merged-forward record QQ renders for us. The indented
# `[发送者]` lines are inside the forward, which is why they survive even though
# the outer message carries no per-element author.
_FORWARD_TRANSCRIPT = """=== 消息 1 ===
[消息内容] 唐完了

=== 消息 2 ===
[消息内容] [群聊的聊天记录]
[消息类型] 合并转发消息
[关联消息]
--- 第1条 ---
    [消息内容] 什么时候来着
    [发送者] 旺旺雪饼
--- 第2条 ---
    [消息内容] 1029
    [发送者] 溯流光

=== 消息 3 ===
[消息内容] 别急，我在调试"""

# Image URLs come from /download with no extension and carry a short-lived rkey.
_IMAGE_TRANSCRIPT = """=== 消息 1 ===
[消息内容] 呼哈哈哈哈！

=== 消息 2 ===
[消息内容] <faceType=6,faceId="0",ext="eyJ0ZXh0IjoiIn0=">
[附件1] 类型:图片 文件名:62602CF6852D4CF0DCD0C76495ED2B93.jpg 尺寸:261x264 大小:8.8KB URL:https://multimedia.nt.qq.com.cn/download?appid=1407&fileid=ABC&rkey=SECRET123&spec=0

=== 消息 3 ===
[消息内容] 别急，我在调试"""


def _group_payload(**overrides):
    payload = {
        "id": "MSG1",
        "message_type": 0,
        "timestamp": "2026-10-09T10:38:56+08:00",
        "content": " 看这个",
        "group_openid": "G1",
        "author": {"member_openid": "U1", "username": "小明"},
    }
    payload.update(overrides)
    return payload


def _transcript_payload(text: str, **overrides):
    """A payload shaped like the real ones: one element holding a transcript."""
    return _group_payload(msg_elements=[{"content": text}], **overrides)


def test_transcript_is_extracted_verbatim():
    rich = flatten_group_payload(_transcript_payload(_FORWARD_TRANSCRIPT))

    assert rich.text == _FORWARD_TRANSCRIPT
    assert rich.has_rich is True
    assert rich.is_empty is False


def test_merged_forward_arrives_pre_flattened():
    """QQ already renders the chat record; we must not re-flatten it.

    The official bot only receives a group message when @-mentioned, so a
    forwarded card never arrives as its own event. The transcript is the only
    path by which it reaches the archive, which is why this assertion matters.
    """
    rich = flatten_group_payload(_transcript_payload(_FORWARD_TRANSCRIPT))

    assert "[群聊的聊天记录]" in rich.text
    assert "[消息类型] 合并转发消息" in rich.text
    assert "[发送者] 旺旺雪饼" in rich.text
    assert "--- 第2条 ---" in rich.text


def test_split_transcript_numbers_sections_and_keeps_bodies_verbatim():
    segments = split_transcript(_FORWARD_TRANSCRIPT)

    assert [segment.index for segment in segments] == [1, 2, 3]
    assert segments[0].content == "[消息内容] 唐完了"
    # The forward's own body survives intact, indentation and all.
    assert "    [发送者] 溯流光" in segments[1].content
    assert segments[2].content == "[消息内容] 别急，我在调试"


def test_split_transcript_without_markers_yields_nothing():
    """No markers means nothing was quoted -- not one giant segment."""
    assert split_transcript("就是一段普通的话，没有分节标记") == ()
    assert split_transcript("") == ()


def test_image_url_is_extracted_from_transcript():
    (ref,) = extract_media_refs(_IMAGE_TRANSCRIPT)

    assert isinstance(ref, MediaRef)
    assert ref.url == (
        "https://multimedia.nt.qq.com.cn/download?appid=1407&fileid=ABC&rkey=SECRET123&spec=0"
    )
    assert ref.filename == "62602CF6852D4CF0DCD0C76495ED2B93.jpg"


def test_attachment_lines_without_the_image_marker_are_skipped():
    """QQ serves images from /download with no extension, so the 类型 marker is
    the only signal -- a voice clip on the same line shape must not be fetched."""
    transcript = (
        "[附件1] 类型:语音 文件名:v.silk 大小:3KB URL:https://multimedia.nt.qq.com.cn/d?a=1\n"
        "[附件2] 类型:文件 文件名:report.pdf URL:https://multimedia.nt.qq.com.cn/d?a=2\n"
    )

    assert extract_media_refs(transcript) == ()


def test_duplicate_image_lines_are_deduplicated():
    line = "[附件1] 类型:图片 文件名:a.jpg URL:https://m.qq.com/d?rkey=S"
    transcript = f"{line}\n{line.replace('[附件1]', '[附件2]')}\n"

    assert len(extract_media_refs(transcript)) == 1


def test_bare_mention_without_transcript_is_empty():
    """A real capture had `msg_elements: null`; consumers must tolerate it."""
    rich = flatten_group_payload(_group_payload(msg_elements=None))

    assert rich.text == ""
    assert rich.segments == ()
    assert rich.media == ()
    assert rich.has_rich is False
    assert rich.is_empty is True


def test_message_scene_ext_is_ignored_as_credentials():
    """`message_scene.ext` holds quote credentials, not recent messages.

    The earlier design read it as the recent-message list and rendered nothing,
    because it is a list of flat strings rather than a mapping.
    """
    rich = flatten_group_payload(
        _group_payload(
            message_scene={
                "ext": [
                    "msg_idx=REFIDX_ohYtVFaRpnRGXmxD",
                    "auth_token=HhQPZ6NlY6OQvoXFF5d9",
                    "ref_msg_idx=TMP_e940352b-d898-4650-b59f-f227330107a0",
                ]
            }
        )
    )

    assert rich.text == ""
    assert rich.has_rich is False


def test_redaction_strips_credentials_recursively():
    redacted = redact_payload({
        "auth_token": "abc",
        "msg_idx": "REFIDX_x",
        "nested": {"ref_msg_idx": "TMP_y", "url": "https://m.qq.com/d?rkey=SECRET&a=1"},
        "list": [{"token": "t"}],
    })

    assert redacted["auth_token"] == "<redacted>"
    assert redacted["msg_idx"] == "<redacted>"
    assert redacted["nested"]["ref_msg_idx"] == "<redacted>"
    assert "SECRET" not in redacted["nested"]["url"]
    assert redacted["list"][0]["token"] == "<redacted>"


def test_payload_carries_redacted_raw_but_live_media_urls():
    """The archive keeps a redacted copy; the fetch keeps the live rkey.

    These must not be the same object: the rkey expires within minutes, so the
    URL is used once and never persisted, while the raw payload is written to
    disk.
    """
    rich = flatten_group_payload(_transcript_payload(_IMAGE_TRANSCRIPT))

    assert rich.raw is not None
    assert "SECRET123" not in str(rich.raw)
    assert "SECRET123" in rich.media_urls[0]


def test_message_ids_are_collected_for_dedup():
    assert flatten_group_payload(_transcript_payload("x")).message_ids == ("MSG1",)
    assert flatten_group_payload(_group_payload(id=None)).message_ids == ()


def test_non_mapping_input_degrades_quietly():
    for value in (None, "not a payload", 42):
        rich = flatten_group_payload(value)  # type: ignore[arg-type]
        assert isinstance(rich, RichContent)
        assert rich.is_empty is True
        assert rich.has_rich is False


def test_census_reports_the_measured_fields():
    line = census(_transcript_payload(_IMAGE_TRANSCRIPT, message_type=103))

    assert "msg_id='MSG1'" in line
    assert "message_type=103" in line
    assert "n_msg_elements=1" in line
    assert "n_segments=3" in line
    assert "n_images=1" in line
    assert "has_ark=False" in line
    assert "n_attachments=0" in line
    assert "author_keys=['member_openid', 'username']" in line
    assert "timestamp='2026-10-09T10:38:56+08:00'" in line


def test_census_counts_scene_ext_without_reading_it_as_content():
    line = census(_group_payload(message_scene={"ext": ["a", "b"]}))

    assert "n_scene_ext=2" in line
    assert "transcript_len=0" in line


def test_census_handles_absent_payload():
    assert census(None) == "payload=absent"


def test_long_transcript_is_truncated():
    """A pathological payload must not blow up the prompt."""
    text = "\n".join(f"=== 消息 {i} ===\n[消息内容] {'x' * 200}" for i in range(1, 200))

    assert len(flatten_group_payload(_transcript_payload(text)).text) <= 8000


def test_safe_attachment_name_is_deterministic_and_sanitised():
    name = safe_attachment_name("MSG/1", 2, "../../etc/passwd")

    assert name == "MSG_1_2_passwd"
    assert "/" not in name
    assert safe_attachment_name("MSG1", 1, "") == "MSG1_1_image.jpg"


def test_rewrite_media_links_replaces_the_url_with_the_saved_path():
    url = extract_media_refs(_IMAGE_TRANSCRIPT)[0].url

    rewritten = rewrite_media_links(
        _IMAGE_TRANSCRIPT,
        {url: "/workspace/attachments/qq/MSG1_1_a.jpg"},
    )

    assert "/workspace/attachments/qq/MSG1_1_a.jpg" in rewritten
    # The rkey dies within minutes; persisting it would archive a dead link.
    assert "rkey" not in rewritten
    assert "SECRET123" not in rewritten
    # Everything else on the line survives.
    assert "文件名:62602CF6852D4CF0DCD0C76495ED2B93.jpg" in rewritten
    assert "尺寸:261x264" in rewritten


def test_rewrite_media_links_marks_a_failed_download():
    rewritten = rewrite_media_links(_IMAGE_TRANSCRIPT, {})

    assert "[图片下载失败]" in rewritten
    assert "multimedia.nt.qq.com.cn" not in rewritten


def test_rewrite_media_links_leaves_other_attachment_kinds_alone():
    """A voice clip's URL was never downloaded, so it must not be rewritten."""
    transcript = "[附件1] 类型:语音 文件名:v.silk URL:https://m.qq.com/d?a=1\n"

    assert rewrite_media_links(transcript, {}) == transcript


def test_rewrite_media_links_covers_repeated_lines():
    url = "https://m.qq.com/d?rkey=S"
    transcript = (
        f"[附件1] 类型:图片 文件名:a.jpg URL:{url}\n"
        f"[附件2] 类型:图片 文件名:a.jpg URL:{url}\n"
    )

    rewritten = rewrite_media_links(transcript, {url: "/tmp/a.jpg"})

    assert rewritten.count("/tmp/a.jpg") == 2
    assert "rkey" not in rewritten


def test_rewrite_media_links_handles_an_empty_transcript():
    assert rewrite_media_links("", {"x": "y"}) == ""
