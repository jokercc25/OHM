"""Tests for compatibility helpers used by channel implementations."""

from __future__ import annotations

import pytest

from openharness.utils.helpers import (
    get_data_path,
    image_extension_for,
    safe_filename,
    sniff_image_type,
    split_message,
)


def test_split_message_prefers_word_boundaries() -> None:
    chunks = split_message("hello world again", 8)

    assert chunks == ["hello", "world", "again"]
    assert all(len(chunk) <= 8 for chunk in chunks)


def test_split_message_hard_splits_long_unbroken_text() -> None:
    chunks = split_message("abcdef", 2)

    assert chunks == ["ab", "cd", "ef"]


def test_split_message_empty_text_returns_no_chunks() -> None:
    assert split_message("", 10) == []


def test_safe_filename_strips_path_and_unsafe_characters() -> None:
    assert safe_filename("../bad name;$(rm).txt") == "bad_name_rm_.txt"


def test_safe_filename_rejects_empty_or_parent_segments() -> None:
    assert safe_filename("../") == ""
    assert safe_filename("...") == ""


def test_get_data_path_is_backwards_compatible_alias() -> None:
    assert get_data_path().name == "data"


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (b"\xff\xd8\xff\xe0" + b"\x00" * 8, "image/jpeg"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, "image/png"),
        (b"GIF87a" + b"\x00" * 8, "image/gif"),
        (b"GIF89a" + b"\x00" * 8, "image/gif"),
        (b"RIFF\x24\x00\x00\x00WEBPVP8 ", "image/webp"),
    ],
)
def test_sniff_image_type_recognises_the_supported_formats(data: bytes, expected: str) -> None:
    assert sniff_image_type(data) == expected


@pytest.mark.parametrize(
    "data",
    [
        b"",  # a CDN answering 200 with nothing
        b"<!DOCTYPE html><html><body>404</body></html>",  # an expired rkey
        b'{"code": 40001, "message": "invalid rkey"}',
        b"\x00\x00\x00\x20ftypavif",  # AVIF: a real image, but unsupported
        b"RIFF\x24\x00\x00\x00AVI LIST",  # RIFF container that is not WebP
        b"\xff\xd8",  # truncated JPEG header
    ],
)
def test_sniff_image_type_rejects_everything_else(data: bytes) -> None:
    assert sniff_image_type(data) is None


def test_image_extension_for_matches_the_sniffed_type() -> None:
    assert image_extension_for("image/jpeg") == ".jpg"
    assert image_extension_for("image/webp") == ".webp"
    assert image_extension_for("application/pdf") == ""
