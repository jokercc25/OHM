"""Tests for the group gossip archive."""

from __future__ import annotations

from pathlib import Path

from openharness.memory.schema import parse_datetime, split_memory_file

from ohmo.gossip import (
    GossipSaveResult,
    build_gossip_index,
    compose_gossip_body,
    gossip_entries_dir,
    gossip_group_dir,
    gossip_index_path,
    list_gossip_headers,
    load_gossip_prompt,
    normalize_gist,
    safe_segment,
    save_gossip_entry,
)
from ohmo.workspace import get_gossip_dir, initialize_workspace

_GROUP = {"channel": "qq", "chat_id": "GROUP_OPENID_1"}
_FORWARD = """=== 消息 2 ===
[消息内容] [群聊的聊天记录]
[消息类型] 合并转发消息
--- 第1条 ---
    [消息内容] 什么时候来着
    [发送者] 旺旺雪饼"""


def _workspace(tmp_path: Path) -> Path:
    return initialize_workspace(tmp_path / ".ohmo")


def _save(workspace: Path, **kwargs: object) -> GossipSaveResult:
    """Save into the default test group unless the test overrides it."""
    kwargs.setdefault("channel", _GROUP["channel"])
    kwargs.setdefault("chat_id", _GROUP["chat_id"])
    return save_gossip_entry(workspace, **kwargs)  # type: ignore[arg-type]


def test_save_writes_entry_with_full_frontmatter(tmp_path: Path):
    workspace = _workspace(tmp_path)

    result = _save(
        workspace,
        raw_text=_FORWARD,
        gist="旺旺雪饼转发：发货时间定在 1029",
        title="发货时间",
        tags=["发货", "小道消息"],
        importance=3,
        sender_member_id="U1",
        sender_display_name="调皮的小男孩",
        created_at="2026-10-09T10:38:56+08:00",
        source_message_ids=["ROBOT1.0_AAA"],
        raw_payload={"id": "ROBOT1.0_AAA", "author": {"username": "调皮的小男孩"}},
    )

    assert result.duplicate is False
    assert result.sensitive is False
    assert result.path.exists()
    assert result.path.parent == gossip_entries_dir(workspace, **_GROUP)

    metadata, body, _, _ = split_memory_file(result.path.read_text(encoding="utf-8"))
    assert metadata["type"] == "reference"
    assert metadata["scope"] == "team"
    assert metadata["category"] == "gossip"
    assert metadata["source"] == "qq_group"
    assert metadata["name"] == "发货时间"
    assert metadata["description"] == "旺旺雪饼转发：发货时间定在 1029"
    assert metadata["gist"] == metadata["description"]
    assert metadata["importance"] == 3
    assert metadata["tags"] == ["发货", "小道消息"]
    assert metadata["channel"] == "qq"
    assert metadata["chat_id"] == "GROUP_OPENID_1"
    assert metadata["sender_display_name"] == "调皮的小男孩"
    assert metadata["msg_ids"] == ["ROBOT1.0_AAA"]
    assert metadata["sensitive"] is False
    assert metadata["signature"]
    # created_at comes from the QQ timestamp, converted to UTC.
    assert parse_datetime(metadata["created_at"]).isoformat() == "2026-10-09T02:38:56+00:00"
    assert "## 原文" in body
    assert "[群聊的聊天记录]" in body


def test_raw_payload_sidecar_is_written_next_to_the_entry(tmp_path: Path):
    workspace = _workspace(tmp_path)

    result = _save(
        workspace,
        raw_text="x",
        gist="g",
        raw_payload={"id": "MSG1", "auth_token": "<redacted>"},
    )

    sidecar = result.path.with_suffix(".raw.json")
    assert sidecar.exists()
    assert "MSG1" in sidecar.read_text(encoding="utf-8")


def test_entries_are_discoverable_by_the_memory_scanner(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _save(workspace, raw_text=_FORWARD, gist="主旨一句话")

    headers = list_gossip_headers(workspace, **_GROUP)

    assert len(headers) == 1
    assert headers[0].title == "主旨一句话"
    assert headers[0].description == "主旨一句话"
    assert headers[0].category == "gossip"
    assert headers[0].memory_type == "reference"


def test_image_placeholders_render_in_the_body(tmp_path: Path):
    workspace = _workspace(tmp_path)

    result = _save(
        workspace,
        raw_text="图片一条",
        gist="带图的瓜",
        media_paths=["/workspace/attachments/qq/IMG_1_a.jpg"],
    )

    _, body, _, _ = split_memory_file(result.path.read_text(encoding="utf-8"))
    assert "/workspace/attachments/qq/IMG_1_a.jpg（待视觉模型补描述）" in body


def test_same_message_id_is_deduped_and_the_body_is_not_rewritten(tmp_path: Path):
    workspace = _workspace(tmp_path)
    first = _save(
        workspace,
        raw_text=_FORWARD,
        gist="第一次",
        source_message_ids=["MSG1"],
    )
    original = first.path.read_text(encoding="utf-8")

    second = _save(
        workspace,
        raw_text=_FORWARD,
        gist="第二次",
        importance=5,
        source_message_ids=["MSG1", "MSG2"],
    )

    assert second.duplicate is True
    assert second.path == first.path
    assert len(list_gossip_headers(workspace, **_GROUP)) == 1

    metadata, body, _, _ = split_memory_file(second.path.read_text(encoding="utf-8"))
    assert metadata["msg_ids"] == ["MSG1", "MSG2"]
    assert metadata["importance"] == 5
    # The gist from the first capture wins: the original is not overwritten.
    assert metadata["description"] == "第一次"
    assert "第一次" in original


def test_identical_text_is_deduped_without_message_ids(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _save(workspace, raw_text=_FORWARD, gist="甲")
    second = _save(workspace, raw_text=_FORWARD, gist="乙")

    assert second.duplicate is True
    assert len(list_gossip_headers(workspace, **_GROUP)) == 1


def test_different_text_saves_twice(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _save(workspace, raw_text="一条瓜", gist="甲")
    _save(workspace, raw_text="另一条瓜", gist="乙")

    assert len(list_gossip_headers(workspace, **_GROUP)) == 2


def test_groups_do_not_share_an_archive(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _save(workspace, raw_text="甲的瓜", gist="甲", channel="qq", chat_id="G1")
    _save(workspace, raw_text="乙的瓜", gist="乙", channel="qq", chat_id="G2")

    assert len(list_gossip_headers(workspace, channel="qq", chat_id="G1")) == 1
    assert len(list_gossip_headers(workspace, channel="qq", chat_id="G2")) == 1
    assert gossip_group_dir(workspace, channel="qq", chat_id="G1") != gossip_group_dir(
        workspace, channel="qq", chat_id="G2"
    )


def test_index_lists_every_entry_and_survives_deletion(tmp_path: Path):
    workspace = _workspace(tmp_path)
    first = _save(workspace, raw_text="瓜一", gist="瓜一", tags=["测试"])
    _save(workspace, raw_text="瓜二", gist="瓜二")

    index_path = gossip_index_path(workspace, **_GROUP)
    index = index_path.read_text(encoding="utf-8")
    assert "- 条数：2" in index
    assert "[瓜一](entries/" in index
    assert "[测试]" in index

    first.path.unlink()
    assert build_gossip_index(workspace, **_GROUP) == index_path
    assert "- 条数：1" in index_path.read_text(encoding="utf-8")


def test_chat_id_cannot_escape_the_gossip_root(tmp_path: Path):
    workspace = _workspace(tmp_path)

    result = _save(workspace, raw_text="瓜", gist="瓜", channel="qq", chat_id="../../etc")

    gossip_root = get_gossip_dir(workspace).resolve()
    assert result.path.resolve().is_relative_to(gossip_root)
    assert safe_segment("../../etc") == "etc"


def test_gist_falls_back_to_the_first_line_of_the_original(tmp_path: Path):
    """The fallback skips the transcript furniture, not just the first line.

    `=== 消息 2 ===` is the literal first non-empty line here, and it would make
    a useless gist -- which is exactly what the fallback sees when a whole
    transcript is archived instead of one segment.
    """
    workspace = _workspace(tmp_path)

    result = _save(workspace, raw_text=_FORWARD, gist="   ")

    assert result.gist == "[群聊的聊天记录]"


def test_gist_falls_back_to_the_sender_when_there_is_no_text(tmp_path: Path):
    workspace = _workspace(tmp_path)

    result = _save(
        workspace,
        raw_text="",
        gist=None,
        media_paths=["/workspace/attachments/qq/a.jpg"],
        sender_display_name="旺旺雪饼",
    )

    assert result.gist == "旺旺雪饼转发的内容"


def test_long_gist_is_truncated(tmp_path: Path):
    workspace = _workspace(tmp_path)

    result = _save(workspace, raw_text="瓜", gist="字" * 200)

    assert len(result.gist) <= 41
    assert result.gist.endswith("…")


def test_secrets_are_marked_not_rejected(tmp_path: Path):
    workspace = _workspace(tmp_path)

    result = _save(
        workspace,
        raw_text="sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        gist="一条带密钥的瓜",
    )

    assert result.sensitive is True
    assert result.path.exists()
    metadata, _, _, _ = split_memory_file(result.path.read_text(encoding="utf-8"))
    assert metadata["sensitive"] is True


def test_body_keeps_the_original_verbatim():
    body = compose_gossip_body(_FORWARD)

    assert "[发送者] 旺旺雪饼" in body
    assert body.startswith("## 原文")


def test_gist_normalisation_strips_quotes_and_collapses_whitespace():
    assert normalize_gist('  「  甲   乙 」 ', "原文") == "甲 乙"
    assert normalize_gist("", "第一行\n第二行") == "第一行"


def test_prompt_reports_counts_without_leaking_gists(tmp_path: Path):
    workspace = _workspace(tmp_path)
    _save(workspace, raw_text="瓜", gist="不该出现在提示词里的主旨")

    prompt = load_gossip_prompt(workspace)

    assert prompt is not None
    assert "屎库" in prompt
    assert "ohmo_save_gossip" in prompt
    assert "共 1 个群、1 条" in prompt
    assert "不该出现在提示词里的主旨" not in prompt


def test_prompt_renders_on_an_empty_workspace(tmp_path: Path):
    workspace = _workspace(tmp_path)

    prompt = load_gossip_prompt(workspace)

    assert prompt is not None
    assert "共 0 个群、0 条" in prompt
