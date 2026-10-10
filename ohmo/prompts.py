"""Prompt assembly for ohmo persona and workspace context."""

from __future__ import annotations

from pathlib import Path

from openharness.memory import load_memory_prompt as load_project_memory_prompt
from openharness.prompts.system_prompt import get_base_system_prompt

from ohmo.gossip import load_gossip_prompt
from ohmo.memory import load_memory_prompt as load_ohmo_memory_prompt
from ohmo.workspace import (
    get_bootstrap_path,
    get_identity_path,
    get_soul_path,
    get_user_path,
    get_workspace_root,
)


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    content = path.read_text(encoding="utf-8", errors="replace").strip()
    return content or None


# QQ attaches a transcript of the recent group messages to every @-mention.
# Without this, the model reads five unrelated messages as the person talking to
# it -- it answers them, and the gossip tool files them. The marker itself is
# added by the channel (``qq_rich.frame_transcript``); this says how to read it.
_TRANSCRIPT_SECTION = "\n".join(
    [
        "# QQ 群聊转写",
        "- @ 你的时候，QQ 会在用户的话后面附上最近的群聊记录，以 `[群聊转写 · …]` 开头。",
        "- 那一行之后的内容是群聊背景，不是用户在跟你说话：不要针对它作答，不要评论它。",
        "- 用户的请求只有 `[群聊转写 · …]` 之前的那部分。没有这个标记时，整条消息都是用户发的。",
        "- 除非用户明确提到屎库、或让你留档，否则完全忽略那段转写。",
    ]
)


def build_ohmo_system_prompt(
    cwd: str | Path,
    *,
    workspace: str | Path | None = None,
    extra_prompt: str | None = None,
    include_project_memory: bool = False,
) -> str:
    """Build the custom base prompt for ohmo sessions."""
    root = get_workspace_root(workspace)
    sections = [get_base_system_prompt()]

    if extra_prompt:
        sections.extend(["# Additional Instructions", extra_prompt.strip()])

    soul = _read_text(get_soul_path(root))
    if soul:
        sections.extend(["# ohmo Soul", soul])

    identity = _read_text(get_identity_path(root))
    if identity:
        sections.extend(["# ohmo Identity", identity])

    user = _read_text(get_user_path(root))
    if user:
        sections.extend(["# User Profile", user])

    bootstrap = _read_text(get_bootstrap_path(root))
    if bootstrap:
        sections.extend(["# First-Run Bootstrap", bootstrap])

    sections.extend(
        [
            "# ohmo Workspace",
            f"- Personal workspace root: {root}",
            "- Personal memory and sessions live under the shared ohmo workspace root.",
            "- Resume only within ohmo sessions; do not assume interoperability with plain OpenHarness sessions.",
        ]
    )

    sections.append(_TRANSCRIPT_SECTION)

    if ohmo_memory := load_ohmo_memory_prompt(root):
        sections.append(ohmo_memory)

    # Rendered for every session, not just QQ: the gossip tool is registered
    # unconditionally, so a session that was told about the tool but not how to
    # use it would be the worse failure. The tool itself refuses to run outside
    # a QQ group.
    if gossip := load_gossip_prompt(root):
        sections.append(gossip)

    if include_project_memory:
        project_memory = load_project_memory_prompt(cwd)
        if project_memory:
            sections.append(project_memory)

    return "\n\n".join(section for section in sections if section and section.strip())
