"""Tests for QQ rich-payload flattening.

This file must not import ``botpy``: ``qq_rich`` is deliberately SDK-free so the
flattening logic (the part most likely to be wrong) stays testable everywhere.
"""

from __future__ import annotations

from openharness.channels.impl.qq_rich import (
    RichContent,
    census,
    flatten_group_payload,
    is_downloadable_image,
    redact_payload,
    safe_attachment_name,
)


def _group_payload(**overrides):
    payload = {
        "id": "MSG1",
        "message_type": 0,
        "timestamp": 1759900000,
        "content": "看这个",
        "group_openid": "G1",
        "author": {"member_openid": "U1", "username": "小明"},
    }
    payload.update(overrides)
    return payload


def test_plain_text_element_is_rendered_with_sender():
    payload = _group_payload(
        msg_elements=[
            {"message_type": 0, "id": "E1", "content": "昨天的瓜",
             "author": {"username": "小红"}},
        ]
    )

    rich = flatten_group_payload(payload)

    assert "小红: 昨天的瓜" in rich.text
    assert rich.has_rich is True


def test_chat_record_renders_as_quoted_transcript():
    payload = _group_payload(
        msg_elements=[
            {
                "message_type": 102,
                "id": "E2",
                "author": {"username": "小明"},
                "msg_elements": [
                    {"message_type": 0, "id": "E3", "content": "真的要降价了",
                     "author": {"username": "小刚"}},
                    {"message_type": 0, "id": "E4", "content": "链接在此",
                     "author": {"username": "小刚"}},
                ],
            }
        ]
    )

    rich = flatten_group_payload(payload)

    assert "[转发聊天记录]" in rich.text
    assert "> 小刚: 真的要降价了" in rich.text
    assert "> 小刚: 链接在此" in rich.text
    # Nested ids are collected so a re-forward dedups against the same record.
    assert set(rich.message_ids) == {"MSG1", "E2", "E3", "E4"}


def test_nested_elements_are_not_rendered_twice():
    """Top level renders only direct elements; nesting stays inside the record."""
    payload = _group_payload(
        msg_elements=[
            {
                "message_type": 102,
                "id": "E2",
                "author": {"username": "小明"},
                "msg_elements": [
                    {"message_type": 0, "id": "E3", "content": "只应出现一次",
                     "author": {"username": "小刚"}},
                ],
            }
        ]
    )

    assert flatten_group_payload(payload).text.count("只应出现一次") == 1


def test_ark_card_is_rendered_with_link():
    payload = _group_payload(
        msg_elements=[
            {
                "message_type": 3,
                "id": "E2",
                "author": {"username": "小明"},
                "ark_data": {
                    "ark_name": "图文H5",
                    "kv": {
                        "title": "显卡要降价",
                        "desc": "传某厂清库存",
                        "jump_url": "https://xiaoheihe.cn/p/1",
                    },
                },
            }
        ]
    )

    rich = flatten_group_payload(payload)

    assert "[链接卡片: 图文H5]" in rich.text
    assert "title: 显卡要降价" in rich.text
    assert "jump_url: https://xiaoheihe.cn/p/1" in rich.text


def test_top_level_ark_is_rendered():
    payload = _group_payload(
        ark_data={"ark_name": "feed", "kv": {"title": "T", "jump_url": "https://u"}}
    )

    rich = flatten_group_payload(payload)

    assert "title: T" in rich.text
    assert rich.has_rich is True


def test_quote_element_references_previous_message():
    payload = _group_payload(
        msg_elements=[
            {
                "message_type": 103,
                "id": "E2",
                "author": {"username": "小明"},
                "message_reference": {"content": "被引用的原话"},
            }
        ]
    )

    assert "> 引用: 被引用的原话" in flatten_group_payload(payload).text


def test_image_attachment_is_collected_but_voice_is_only_noted():
    payload = _group_payload(
        attachments=[
            {"content_type": "image/jpeg", "filename": "a.jpg", "size": 10,
             "url": "https://multimedia.nt.qq.com.cn/download?rkey=SECRET&x=1"},
            {"content_type": "voice", "filename": "v.silk", "url": "https://x/v"},
        ]
    )

    rich = flatten_group_payload(payload)

    assert rich.media_urls == ("https://multimedia.nt.qq.com.cn/download?rkey=SECRET&x=1",)
    assert "[图片: a.jpg]" in rich.text
    assert "[附件: v.silk (voice)]" in rich.text


def test_duplicate_attachment_urls_are_deduplicated():
    attachment = {"content_type": "image/png", "filename": "a.png", "url": "https://x/a.png"}

    rich = flatten_group_payload(_group_payload(attachments=[attachment, dict(attachment)]))

    assert rich.media_urls == ("https://x/a.png",)


def test_card_only_message_is_not_empty():
    """A forwarded card carries no text; it must still count as content.

    qq._on_group_message returns early when the flattened content is empty, so
    this is the guard that keeps card-only messages from being dropped.
    """
    payload = _group_payload(
        content=None,
        msg_elements=[
            {"message_type": 3, "id": "E2",
             "ark_data": {"ark_name": "feed", "kv": {"title": "T", "jump_url": "https://u"}}},
        ],
    )

    rich = flatten_group_payload(payload)

    assert rich.has_rich is True
    assert rich.is_empty is False


def test_unknown_element_type_is_marked_not_dropped():
    rich = flatten_group_payload(_group_payload(msg_elements=[{"message_type": 999}]))

    assert "未知元素 type=999" in rich.text


def test_message_scene_ext_is_rendered_as_recent_context():
    payload = _group_payload(
        message_scene={"ext": {"msg_elements": [
            {"message_type": 0, "content": "上文一句", "author": {"username": "小红"}},
        ]}}
    )

    assert "> 小红: 上文一句" in flatten_group_payload(payload).text


def test_element_walk_is_bounded():
    """A pathological payload must not explode the prompt."""
    deep = {"message_type": 0, "content": "deep", "author": {"username": "x"}}
    for _ in range(6):
        deep = {"message_type": 102, "msg_elements": [deep], "author": {"username": "x"}}

    assert len(flatten_group_payload(_group_payload(msg_elements=[deep])).text) < 2000


def test_redaction_strips_credentials_recursively():
    redacted = redact_payload({
        "auth_token": "abc",
        "msg_idx": 7,
        "nested": {"ref_msg_idx": 9, "url": "https://m.qq.com/d?rkey=SECRET&a=1"},
        "list": [{"token": "t"}],
    })

    assert redacted["auth_token"] == "<redacted>"
    assert redacted["msg_idx"] == "<redacted>"
    assert redacted["nested"]["ref_msg_idx"] == "<redacted>"
    assert "SECRET" not in redacted["nested"]["url"]
    assert redacted["list"][0]["token"] == "<redacted>"


def test_flatten_redacts_the_payload_it_carries():
    rich = flatten_group_payload(
        _group_payload(
            attachments=[{"content_type": "image/png", "filename": "a.png",
                          "url": "https://m.qq.com/d?rkey=SECRET"}]
        )
    )

    assert rich.raw is not None
    assert "SECRET" not in str(rich.raw)
    # The download URL keeps its rkey: it is used immediately, never persisted.
    assert "SECRET" in rich.media_urls[0]


def test_non_mapping_input_degrades_quietly():
    for value in (None, "not a payload", 42):
        rich = flatten_group_payload(value)  # type: ignore[arg-type]
        assert isinstance(rich, RichContent)
        assert rich.is_empty is True
        assert rich.has_rich is False


def test_census_reports_the_phase_zero_signals():
    line = census(
        _group_payload(
            msg_elements=[{"message_type": 102, "msg_elements": [
                {"message_type": 0, "content": "x"}]}],
            ark_data={"ark_name": "feed"},
            attachments=[{"content_type": "image/png", "url": "https://u"}],
            message_scene={"ext": {"msg_elements": [{"message_type": 0}]}},
        )
    )

    assert "msg_id='MSG1'" in line
    assert "n_msg_elements=1" in line
    assert "first_elem_type=102" in line
    assert "has_ark=True" in line
    assert "n_attachments=1" in line
    assert "has_scene_ext=True" in line
    assert "author_keys=['member_openid', 'username']" in line


def test_census_handles_absent_payload():
    assert census(None) == "payload=absent"


def test_safe_attachment_name_is_deterministic_and_sanitised():
    name = safe_attachment_name("MSG/1", 2, "../../etc/passwd")

    assert name == "MSG_1_2_passwd"
    assert "/" not in name
    assert safe_attachment_name("MSG1", 1, "") == "MSG1_1_image.jpg"


def test_is_downloadable_image_by_type_and_extension():
    assert is_downloadable_image({"content_type": "image/png"}) is True
    assert is_downloadable_image({"filename": "b.WEBP"}) is True
    assert is_downloadable_image({"filename": "v.silk"}) is False
    assert is_downloadable_image({}) is False
