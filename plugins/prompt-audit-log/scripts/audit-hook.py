#!/usr/bin/env python3
"""
Cursor hook: records the datetime BEFORE the agent writes or edits a file.

  beforeSubmitPrompt -> logs the submitted prompt
  preToolUse         -> fires before ANY tool executes. Filtered here to the
                        file-writing tools. This is the actual pre-write
                        timestamp.
  postToolUse        -> paired against the above for completion time.

CRITICAL: exit code 2 from a preToolUse hook DENIES the tool call in Cursor.
This script must therefore always exit 0, on every path, or it would silently
block the agent from editing files. See main().
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Which tool names count as "writing a file". Cursor's exact tool names vary by
# version, so this is a regex and it is overridable. See the discovery note in
# the README: tools-seen.json records every tool name observed, so you can
# tighten this to what your Cursor build actually emits.
WRITE_TOOLS = re.compile(
    os.environ.get(
        "CURSOR_AUDIT_WRITE_TOOLS",
        r"^(Write|Edit|MultiEdit|SearchReplace|StrReplace|Create|CreateFile|"
        r"DeleteFile|TabWrite|ApplyPatch|NotebookEdit)",
    ),
    re.IGNORECASE,
)

# tool_input key holding the target path differs across tools/versions.
PATH_KEYS = (
    "file_path",
    "filePath",
    "path",
    "target_file",
    "targetFile",
    "relative_workspace_path",
    "notebook_path",
)

MAX_PROMPT_CHARS = int(os.environ.get("CURSOR_AUDIT_MAX_PROMPT_CHARS", "0"))
STATE_TTL_SECONDS = 24 * 60 * 60


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def log_dir(payload: dict) -> Path:
    override = os.environ.get("CURSOR_AUDIT_DIR")
    if override:
        return Path(override).expanduser()
    roots = payload.get("workspace_roots") or []
    if roots:
        return Path(roots[0]) / ".cursor-audit"
    return Path.home() / ".cursor" / "audit"


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def target_path(tool_input: object) -> str | None:
    if not isinstance(tool_input, dict):
        return None
    for key in PATH_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def resolve(file_path: str, payload: dict) -> Path:
    """Resolve a possibly-relative tool path against cwd or the workspace root."""
    path = Path(file_path)
    if path.is_absolute():
        return path
    base = payload.get("cwd") or ""
    if not base:
        roots = payload.get("workspace_roots") or []
        base = roots[0] if roots else "."
    return Path(base) / path


def common(payload: dict) -> dict:
    return {
        "conversation_id": payload.get("conversation_id"),
        "generation_id": payload.get("generation_id"),
        "session_id": payload.get("session_id"),
        "cursor_version": payload.get("cursor_version"),
    }


# --- pairing state -----------------------------------------------------------

def pending_path(out_dir: Path, payload: dict, tool_name: str, file_path: str) -> Path:
    import hashlib

    key = f"{payload.get('conversation_id')}::{tool_name}::{file_path}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return out_dir / "state" / f"{digest}.json"


def prune_state(out_dir: Path) -> None:
    root = out_dir / "state"
    if not root.is_dir():
        return
    cutoff = time.time() - STATE_TTL_SECONDS
    for path in root.glob("*.json"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def note_tool_name(out_dir: Path, tool_name: str, matched: bool) -> None:
    """
    Accumulate observed tool names so the WRITE_TOOLS regex can be tightened to
    whatever this Cursor build actually emits.
    """
    path = out_dir / "tools-seen.json"
    try:
        seen = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        seen = {}
    entry = seen.setdefault(tool_name, {"count": 0, "treated_as_write": matched})
    entry["count"] += 1
    entry["treated_as_write"] = matched
    try:
        write_atomic(path, json.dumps(seen, indent=2, sort_keys=True) + "\n")
    except OSError:
        pass


# --- handlers ----------------------------------------------------------------

def handle_prompt(payload: dict, out_dir: Path) -> None:
    prompt = payload.get("prompt", "")
    stored, truncated = prompt, False
    if MAX_PROMPT_CHARS and len(prompt) > MAX_PROMPT_CHARS:
        stored, truncated = prompt[:MAX_PROMPT_CHARS], True

    record = {
        "datetime": now_iso(),
        "event": "beforeSubmitPrompt",
        **common(payload),
        "prompt": stored,
        "prompt_chars": len(prompt),
        "attachments": [
            a.get("file_path")
            for a in payload.get("attachments", [])
            if isinstance(a, dict)
        ],
    }
    if truncated:
        record["truncated"] = True

    append_jsonl(out_dir / "prompts.jsonl", record)
    prune_state(out_dir)


def handle_pre_tool(payload: dict, out_dir: Path) -> None:
    """The pre-write timestamp. Fires before the tool runs."""
    tool_name = payload.get("tool_name") or "unknown"
    is_write = bool(WRITE_TOOLS.match(tool_name))
    note_tool_name(out_dir, tool_name, is_write)

    if not is_write:
        return

    stamp = now_iso()
    tool_input = payload.get("tool_input")
    file_path = target_path(tool_input)

    # Only knowable because this fires BEFORE the write.
    existed_before = None
    if file_path:
        try:
            existed_before = resolve(file_path, payload).exists()
        except OSError:
            pass

    record = {
        "datetime": stamp,
        "event": "preToolUse",
        "tool_name": tool_name,
        "file_path": file_path,
        "existed_before": existed_before,
        **common(payload),
    }
    append_jsonl(out_dir / "write-starts.jsonl", record)
    write_atomic(
        out_dir / "writes-started.txt",
        f"{stamp}\t{tool_name}\t{file_path or '(no path in tool_input)'}\n",
    )

    if file_path:
        write_atomic(
            pending_path(out_dir, payload, tool_name, file_path),
            json.dumps({"started_at": stamp, "existed_before": existed_before}),
        )


def handle_post_tool(payload: dict, out_dir: Path) -> None:
    tool_name = payload.get("tool_name") or "unknown"
    if not WRITE_TOOLS.match(tool_name):
        return

    finished_at = now_iso()
    file_path = target_path(payload.get("tool_input"))
    started_at = None
    existed_before = None

    if file_path:
        state_file = pending_path(out_dir, payload, tool_name, file_path)
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
            started_at = state.get("started_at")
            existed_before = state.get("existed_before")
            state_file.unlink()
        except (OSError, json.JSONDecodeError):
            pass

    record = {
        "started_at": started_at,
        "finished_at": finished_at,
        "event": "postToolUse",
        "tool_name": tool_name,
        "file_path": file_path,
        "existed_before": existed_before,
        **common(payload),
    }
    if started_at:
        try:
            delta = datetime.fromisoformat(finished_at) - datetime.fromisoformat(
                started_at
            )
            record["elapsed_ms"] = int(delta.total_seconds() * 1000)
        except ValueError:
            pass

    append_jsonl(out_dir / "write-events.jsonl", record)


# --- entrypoint --------------------------------------------------------------

HANDLERS = {
    "beforeSubmitPrompt": handle_prompt,
    "preToolUse": handle_pre_tool,
    "postToolUse": handle_post_tool,
}


def respond(event: str | None) -> None:
    """Always permissive. preToolUse denies on exit code 2, so never exit 2."""
    if event == "beforeSubmitPrompt":
        print(json.dumps({"continue": True}))
    elif event == "preToolUse":
        print(json.dumps({"permission": "allow"}))
    # postToolUse expects no response.


def main() -> None:
    event = None
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        event = payload.get("hook_event_name")

        handler = HANDLERS.get(event)
        if handler:
            handler(payload, log_dir(payload))
        else:
            print(f"[prompt-audit-log] ignoring event: {event}", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"[prompt-audit-log] {event} failed: {exc}", file=sys.stderr)

    try:
        respond(event)
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
    # Explicit: never propagate a non-zero status. Exit 2 would deny the tool.
    sys.exit(0)
