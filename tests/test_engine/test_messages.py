from __future__ import annotations

import base64
from pathlib import Path

import pytest

from openharness.engine.messages import (
    ConversationMessage,
    ImageBlock,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    sanitize_conversation_messages,
)

_PNG = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    b"\x90wS\xde\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01"
    b"\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_sanitize_conversation_messages_keeps_complete_tool_turn():
    messages = [
        ConversationMessage.from_user_text("edit the file"),
        ConversationMessage(
            role="assistant",
            content=[ToolUseBlock(id="write_file:234", name="write_file", input={"path": "x"})],
        ),
        ConversationMessage(
            role="user",
            content=[ToolResultBlock(tool_use_id="write_file:234", content="ok", is_error=False)],
        ),
    ]

    sanitized = sanitize_conversation_messages(messages)

    assert sanitized == messages


def test_sanitize_conversation_messages_drops_dangling_trailing_tool_use():
    messages = [
        ConversationMessage.from_user_text("edit the file"),
        ConversationMessage(
            role="assistant",
            content=[ToolUseBlock(id="write_file:234", name="write_file", input={"path": "x"})],
        ),
    ]

    sanitized = sanitize_conversation_messages(messages)

    assert sanitized == [ConversationMessage.from_user_text("edit the file")]


def test_image_block_from_path_reads_the_type_from_the_bytes(tmp_path: Path):
    """A WebP named .jpg must report itself as a WebP.

    Providers judge the format by content, so a block that lies about its media
    type describes an image that does not exist.
    """
    path = tmp_path / "mislabeled.jpg"
    path.write_bytes(b"RIFF\x24\x00\x00\x00WEBPVP8 " + b"\x00" * 24)

    block = ImageBlock.from_path(path)

    assert block.media_type == "image/webp"
    assert base64.b64decode(block.data) == path.read_bytes()


def test_image_block_from_path_accepts_a_real_png(tmp_path: Path):
    path = tmp_path / "a.png"
    path.write_bytes(_PNG)

    assert ImageBlock.from_path(path).media_type == "image/png"


@pytest.mark.parametrize(
    "payload",
    [
        b"<!DOCTYPE html><html><body>404</body></html>",
        b"",
        b"\x00\x00\x00\x20ftypavif",
    ],
)
def test_image_block_from_path_refuses_a_file_that_is_not_a_supported_image(
    tmp_path: Path, payload: bytes
):
    """Refusing here is what keeps a bad download out of the request body.

    The block is persisted into the session snapshot, so accepting one bad file
    means the provider rejects that same conversation on every later turn.
    """
    path = tmp_path / "a.jpg"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="Unsupported image attachment"):
        ImageBlock.from_path(path)


def test_sanitize_conversation_messages_drops_orphan_tool_results_but_keeps_user_text():
    messages = [
        ConversationMessage.from_user_text("hello"),
        ConversationMessage(
            role="user",
            content=[
                ToolResultBlock(tool_use_id="missing_call", content="stale", is_error=True),
                TextBlock(text="new prompt"),
            ],
        ),
    ]

    sanitized = sanitize_conversation_messages(messages)

    assert sanitized == [
        ConversationMessage.from_user_text("hello"),
        ConversationMessage(role="user", content=[TextBlock(text="new prompt")]),
    ]
