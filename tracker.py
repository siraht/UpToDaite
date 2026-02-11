#!/usr/bin/env python3
"""Project activity tracker for /data/projects/* workspaces.

Features:
- Detect active projects from tmux panes.
- Ingest assistant outputs from Codex and Claude local JSONL session logs.
- Append assistant outputs to <project>/.agent/outputlog.md.
- Summarize unread outputlog content (marker -> EOF) via non-interactive LLM call.
- Append summaries to <project>/.agent/summarylog.md and rewrite .agent/projectstatus.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

MARKER_LINE = "<!-- AGENT_TRACKER_MARKER -->"
OUTPUTLOG_HEADER = """# Agent Output Log

This file stores assistant turn outputs captured from Codex/Claude sessions.
Unread content for summarization is everything after the marker line.

"""
SUMMARYLOG_HEADER = """# Summary Log

This file stores timestamped summaries generated from new output log entries.

"""
DEFAULT_STATUS = """# Project Status Dashboard

Last updated: Unknown

## Latest Event

- No processed events yet.

## Active Threads

- Unknown

## Subthreads

- Unknown

## Ideas Not Started

- Unknown

## Ideas In Progress

- Unknown

## Recently Finished

- Unknown
"""

DEFAULT_PROJECTS_ROOT = Path("/data/projects")
DEFAULT_STATE_PATH = Path.home() / ".agent-tracker" / "state.json"
DEFAULT_CODEX_SESSIONS_DIR = Path.home() / ".codex" / "sessions"
DEFAULT_CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
DEFAULT_CLAUDE_SETTINGS_PATH = Path.home() / ".claude" / "settings-synthetic.json"
DEFAULT_CLAUDE_BETAS = "interleaved-thinking"
DEFAULT_FIRST_SEEN_HISTORY = "tail"
LLM_TIMEOUT_SECONDS = 120
EVENT_ID_PATTERN = re.compile(r"\bevent=(\S+)")


@dataclass
class OutputEvent:
    project_dir: Path
    source: str
    timestamp: str
    session_id: str
    message_id: str
    event_id: str
    text: str


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track and summarize project assistant outputs.")
    parser.add_argument("command", choices=["run", "discover", "ingest", "summarize"], nargs="?", default="run")
    parser.add_argument("--projects-root", default=str(DEFAULT_PROJECTS_ROOT))
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--codex-sessions-dir", default=str(DEFAULT_CODEX_SESSIONS_DIR))
    parser.add_argument("--claude-projects-dir", default=str(DEFAULT_CLAUDE_PROJECTS_DIR))
    parser.add_argument("--project", action="append", default=[], help="Project path (repeatable).")
    parser.add_argument("--all-projects", action="store_true", help="Ignore tmux and process all projects seen in events.")
    parser.add_argument("--initial-tail-bytes", type=int, default=200_000, help="Tail bytes for unseen session files.")
    parser.add_argument(
        "--first-seen-history",
        choices=["full", "tail"],
        default=DEFAULT_FIRST_SEEN_HISTORY,
        help="How to initialize unseen source offsets: full=scan from start, tail=scan recent tail bytes.",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Reset source offsets using --first-seen-history and --initial-tail-bytes policy.",
    )
    parser.add_argument("--engine", default="claude", choices=["claude", "codex"], help="Summarization engine.")
    parser.add_argument(
        "--claude-settings",
        default=str(DEFAULT_CLAUDE_SETTINGS_PATH),
        help="Path to Claude CLI settings file.",
    )
    parser.add_argument(
        "--claude-betas",
        default=DEFAULT_CLAUDE_BETAS,
        help="Value passed to Claude CLI --betas.",
    )
    parser.add_argument(
        "--claude-dangerously-skip-permissions",
        dest="claude_skip_permissions",
        action="store_true",
        default=True,
        help="Pass --dangerously-skip-permissions to Claude CLI.",
    )
    parser.add_argument(
        "--no-claude-dangerously-skip-permissions",
        dest="claude_skip_permissions",
        action="store_false",
        help="Do not pass --dangerously-skip-permissions to Claude CLI.",
    )
    parser.add_argument("--max-unread-chars", type=int, default=40_000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_state(state_file: Path) -> Dict[str, Any]:
    state = read_json(state_file, {"version": 1, "sources": {}, "sessions": {}})
    if not isinstance(state, dict):
        state = {"version": 1, "sources": {}, "sessions": {}}
    state.setdefault("version", 1)
    state.setdefault("sources", {})
    state.setdefault("sessions", {})
    if not isinstance(state["sources"], dict):
        state["sources"] = {}
    if not isinstance(state["sessions"], dict):
        state["sessions"] = {}
    return state


def save_state(state_file: Path, state: Dict[str, Any]) -> None:
    write_json(state_file, state)


def normalize_project_root(candidate: Path, projects_root: Path) -> Optional[Path]:
    try:
        candidate = candidate.resolve()
        projects_root = projects_root.resolve()
    except FileNotFoundError:
        candidate = candidate.absolute()
        projects_root = projects_root.absolute()

    try:
        rel = candidate.relative_to(projects_root)
    except Exception:
        return None

    parts = rel.parts
    if not parts:
        return None
    project = projects_root / parts[0]
    if project.exists() and project.is_dir():
        return project
    return None


def discover_active_projects(projects_root: Path, verbose: bool = False) -> List[Path]:
    cmd = ["tmux", "list-panes", "-a", "-F", "#{session_name}\t#{pane_current_path}\t#{pane_current_command}"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if verbose:
            print("tmux list-panes unavailable; no active project filter from tmux.", file=sys.stderr)
        return []

    projects: Dict[str, Path] = {}
    for line in proc.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        pane_path = parts[1].strip()
        if not pane_path:
            continue
        project = normalize_project_root(Path(pane_path), projects_root)
        if project is not None:
            projects[str(project)] = project
    return sorted(projects.values(), key=lambda p: str(p))


def ensure_source_state(
    state: Dict[str, Any],
    source_path: Path,
    kind: str,
    initial_tail_bytes: int,
    first_seen_history: str,
    reindex: bool,
) -> Dict[str, Any]:
    sources = state["sources"]
    key = str(source_path)
    size = source_path.stat().st_size
    if first_seen_history == "full":
        baseline_offset = 0
    else:
        baseline_offset = max(0, size - max(0, initial_tail_bytes))

    if reindex or key not in sources:
        entry = {
            "kind": kind,
            "offset": baseline_offset,
            "cwd_hint": None,
            "session_id": None,
            "last_seen_ts": None,
            "history_mode": first_seen_history,
            "history_complete": False,
        }
        sources[key] = entry
    else:
        raw_entry = sources.get(key)
        if isinstance(raw_entry, dict):
            entry = raw_entry
        else:
            entry = {
                "kind": kind,
                "offset": baseline_offset,
                "cwd_hint": None,
                "session_id": None,
                "last_seen_ts": None,
                "history_mode": first_seen_history,
                "history_complete": False,
            }
            sources[key] = entry
        entry.setdefault("kind", kind)
        entry.setdefault("offset", 0)
        entry.setdefault("cwd_hint", None)
        entry.setdefault("session_id", None)
        entry.setdefault("last_seen_ts", None)
        entry.setdefault("history_mode", "tail")
        entry.setdefault("history_complete", False)
        try:
            entry["offset"] = int(entry.get("offset", 0) or 0)
        except Exception:
            entry["offset"] = 0
        if first_seen_history == "full" and entry.get("history_mode") != "full" and not entry.get("history_complete"):
            # Upgrade legacy tail-based entries to full-history replay one time.
            entry["offset"] = 0
            entry["history_mode"] = "full"
        if entry["offset"] < 0:
            entry["offset"] = 0
        # If a source file was truncated/rotated, clamp offset back into range.
        if entry["offset"] > size:
            entry["offset"] = baseline_offset
    entry["history_mode"] = "full" if first_seen_history == "full" else str(entry.get("history_mode") or "tail")
    return entry


def update_session_tracking(
    state: Dict[str, Any],
    kind: str,
    source_path: Path,
    source_state: Dict[str, Any],
    events: List[OutputEvent],
    projects_root: Path,
) -> None:
    sessions = state.get("sessions")
    if not isinstance(sessions, dict):
        sessions = {}
        state["sessions"] = sessions

    key = f"{kind}:{source_path}"
    raw_entry = sessions.get(key)
    if not isinstance(raw_entry, dict):
        raw_entry = {}
    entry = raw_entry

    now_ts = now_utc_iso()
    entry["kind"] = kind
    entry["source_path"] = str(source_path)
    entry.setdefault("first_seen_ts", now_ts)
    entry["last_scanned_ts"] = now_ts

    session_id = source_state.get("session_id")
    if isinstance(session_id, str) and session_id:
        entry["session_id"] = session_id

    cwd_hint = source_state.get("cwd_hint")
    if isinstance(cwd_hint, str) and cwd_hint:
        hinted_project = normalize_project_root(Path(cwd_hint), projects_root)
        if hinted_project is not None:
            entry["project_hint"] = str(hinted_project)

    try:
        entry["last_offset"] = int(source_state.get("offset", 0) or 0)
    except Exception:
        entry["last_offset"] = 0

    if events:
        entry["ingested"] = True
        entry.setdefault("first_ingested_ts", now_ts)
        entry["last_ingested_ts"] = now_ts
        entry["last_event_ts"] = str(events[-1].timestamp)
        entry["last_event_id"] = str(events[-1].event_id)

        project_dirs: set[str] = set()
        existing_dirs = entry.get("project_dirs")
        if isinstance(existing_dirs, list):
            project_dirs.update(str(x) for x in existing_dirs if isinstance(x, str))
        for evt in events:
            project_dirs.add(str(evt.project_dir))
        entry["project_dirs"] = sorted(project_dirs)

        try:
            total = int(entry.get("event_count_total", 0) or 0)
        except Exception:
            total = 0
        entry["event_count_total"] = total + len(events)
        entry["last_scan_event_count"] = len(events)
    else:
        entry.setdefault("ingested", False)
        entry["last_scan_event_count"] = 0

    sessions[key] = entry


def hint_source_project(
    kind: str,
    source_path: Path,
    source_state: Dict[str, Any],
    projects_root: Path,
) -> Optional[Path]:
    cwd_hint = source_state.get("cwd_hint")
    if isinstance(cwd_hint, str) and cwd_hint:
        project = normalize_project_root(Path(cwd_hint), projects_root)
        if project is not None:
            return project

    if kind == "claude":
        hinted = extract_claude_project_hint(source_path, projects_root)
        if hinted:
            return normalize_project_root(Path(hinted), projects_root)
        return None

    head_cwd, _ = extract_codex_head_hints(source_path)
    if head_cwd:
        return normalize_project_root(Path(head_cwd), projects_root)
    return None


def iter_jsonl_from_offset(path: Path, start_offset: int) -> Tuple[Iterable[Tuple[int, Dict[str, Any]]], int]:
    try:
        file_size = path.stat().st_size
    except OSError:
        file_size = 0
    if start_offset < 0:
        start_offset = 0
    if start_offset > file_size:
        start_offset = file_size

    def _iter() -> Iterable[Tuple[int, Dict[str, Any]]]:
        with path.open("rb") as fh:
            fh.seek(start_offset)
            if start_offset > 0:
                fh.readline()  # discard partial line
            while True:
                line_start = fh.tell()
                raw = fh.readline()
                if not raw:
                    break
                text = raw.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    obj = json.loads(text)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                yield line_start, obj

    with path.open("rb") as fh:
        fh.seek(start_offset)
        if start_offset > 0:
            fh.readline()
        while fh.readline():
            pass
        end_offset = fh.tell()

    return _iter(), end_offset


def extract_codex_head_hints(source_path: Path) -> Tuple[Optional[str], Optional[str]]:
    try:
        with source_path.open("r", encoding="utf-8") as fh:
            first = fh.readline()
        obj = json.loads(first)
    except Exception:
        return None, None
    if not isinstance(obj, dict):
        return None, None

    if obj.get("type") != "session_meta":
        return None, None
    payload = obj.get("payload", {})
    if not isinstance(payload, dict):
        return None, None
    cwd = payload.get("cwd")
    session_id = payload.get("id")
    return cwd, session_id


def parse_codex_events(
    source_path: Path,
    source_state: Dict[str, Any],
    projects_root: Path,
    project_filter: Optional[set[Path]],
) -> Tuple[List[OutputEvent], int]:
    cwd_hint = source_state.get("cwd_hint")
    session_id = source_state.get("session_id")

    if not cwd_hint or not session_id:
        head_cwd, head_session = extract_codex_head_hints(source_path)
        cwd_hint = cwd_hint or head_cwd
        session_id = session_id or head_session

    events: List[OutputEvent] = []
    start_offset = int(source_state.get("offset", 0) or 0)
    iterator, end_offset = iter_jsonl_from_offset(source_path, start_offset)

    for line_start, obj in iterator:
        obj_type = obj.get("type")

        if obj_type == "session_meta":
            payload = obj.get("payload", {})
            if not isinstance(payload, dict):
                payload = {}
            cwd_hint = payload.get("cwd") or cwd_hint
            session_id = payload.get("id") or session_id
        elif obj_type == "turn_context":
            payload = obj.get("payload", {})
            if not isinstance(payload, dict):
                payload = {}
            cwd_hint = payload.get("cwd") or cwd_hint
            session_id = session_id or payload.get("session_id")

        if obj_type != "response_item":
            continue

        payload = obj.get("payload", {})
        if not isinstance(payload, dict):
            continue
        if payload.get("type") != "message" or payload.get("role") != "assistant":
            continue
        phase = payload.get("phase")
        # Codex event streams may include both commentary updates and final user-facing output.
        # Keep only final answers when phase metadata is available.
        if phase is not None and phase != "final_answer":
            continue

        text_parts: List[str] = []
        content = payload.get("content", [])
        if not isinstance(content, list):
            content = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"output_text", "text"}:
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())

        if not text_parts:
            continue

        if not cwd_hint:
            continue

        project_dir = normalize_project_root(Path(cwd_hint), projects_root)
        if project_dir is None:
            continue
        if project_filter is not None and project_dir not in project_filter:
            continue

        ts = obj.get("timestamp") or now_utc_iso()
        msg_id = payload.get("id") or f"line-{line_start}"
        event_id = f"codex:{source_path}:{line_start}"

        events.append(
            OutputEvent(
                project_dir=project_dir,
                source="codex",
                timestamp=str(ts),
                session_id=session_id or source_path.stem,
                message_id=str(msg_id),
                event_id=event_id,
                text="\n\n".join(text_parts),
            )
        )

    source_state["cwd_hint"] = cwd_hint
    source_state["session_id"] = session_id
    source_state["offset"] = end_offset
    if source_state.get("history_mode") == "full" and start_offset == 0:
        source_state["history_complete"] = True
    source_state["last_seen_ts"] = now_utc_iso()
    return events, end_offset


def extract_claude_project_hint(source_path: Path, projects_root: Path) -> Optional[str]:
    # Path format example: ~/.claude/projects/-data-projects-research/<session>.jsonl
    try:
        projects_idx = source_path.parts.index("projects")
    except ValueError:
        return None
    if projects_idx + 1 >= len(source_path.parts):
        return None

    slug = source_path.parts[projects_idx + 1]
    if not slug.startswith("-data-projects-"):
        return None
    project_name = slug[len("-data-projects-") :]
    candidate = projects_root / project_name
    if candidate.exists() and candidate.is_dir():
        return str(candidate)
    return None


def parse_claude_events(
    source_path: Path,
    source_state: Dict[str, Any],
    projects_root: Path,
    project_filter: Optional[set[Path]],
) -> Tuple[List[OutputEvent], int]:
    cwd_hint = source_state.get("cwd_hint") or extract_claude_project_hint(source_path, projects_root)
    session_id = source_state.get("session_id")

    events: List[OutputEvent] = []
    start_offset = int(source_state.get("offset", 0) or 0)
    iterator, end_offset = iter_jsonl_from_offset(source_path, start_offset)

    for line_start, obj in iterator:
        top_cwd = obj.get("cwd")
        if isinstance(top_cwd, str) and top_cwd:
            cwd_hint = top_cwd

        top_session = obj.get("sessionId")
        if isinstance(top_session, str) and top_session:
            session_id = top_session

        if obj.get("type") != "assistant":
            continue

        message = obj.get("message", {})
        if not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            continue

        text_parts: List[str] = []
        content = message.get("content", [])
        if not isinstance(content, list):
            content = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "text":
                continue
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text.strip())

        if not text_parts:
            continue
        if not cwd_hint:
            continue

        project_dir = normalize_project_root(Path(cwd_hint), projects_root)
        if project_dir is None:
            continue
        if project_filter is not None and project_dir not in project_filter:
            continue

        ts = obj.get("timestamp") or now_utc_iso()
        msg_id = message.get("id") or obj.get("uuid") or f"line-{line_start}"
        event_id = f"claude:{source_path}:{line_start}"

        events.append(
            OutputEvent(
                project_dir=project_dir,
                source="claude",
                timestamp=str(ts),
                session_id=session_id or source_path.stem,
                message_id=str(msg_id),
                event_id=event_id,
                text="\n\n".join(text_parts),
            )
        )

    source_state["cwd_hint"] = cwd_hint
    source_state["session_id"] = session_id
    source_state["offset"] = end_offset
    if source_state.get("history_mode") == "full" and start_offset == 0:
        source_state["history_complete"] = True
    source_state["last_seen_ts"] = now_utc_iso()
    return events, end_offset


def ensure_outputlog(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(OUTPUTLOG_HEADER + MARKER_LINE + "\n", encoding="utf-8")
        return

    text = path.read_text(encoding="utf-8")
    if MARKER_LINE not in text:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + MARKER_LINE + "\n"
        path.write_text(text, encoding="utf-8")


def format_output_event(event: OutputEvent) -> str:
    clean_text = event.text.rstrip()
    header = (
        f"### [{event.timestamp}] source={event.source} "
        f"session={event.session_id} message={event.message_id} event={event.event_id}"
    )
    return f"{header}\n\n{clean_text}\n\n"


def read_existing_event_ids(outputlog: Path) -> set[str]:
    if not outputlog.exists():
        return set()
    text = outputlog.read_text(encoding="utf-8")
    return {m.group(1) for m in EVENT_ID_PATTERN.finditer(text)}


def append_output_events(project_dir: Path, events: List[OutputEvent], dry_run: bool = False) -> int:
    if not events:
        return 0

    outputlog = project_dir / ".agent" / "outputlog.md"
    existing_ids = read_existing_event_ids(outputlog)
    unique_events: List[OutputEvent] = []
    seen_ids = set(existing_ids)
    for evt in events:
        if evt.event_id in seen_ids:
            continue
        seen_ids.add(evt.event_id)
        unique_events.append(evt)

    if not unique_events:
        return 0

    if dry_run:
        return len(unique_events)

    ensure_outputlog(outputlog)

    payload = "".join(format_output_event(evt) for evt in unique_events)
    with outputlog.open("a", encoding="utf-8") as fh:
        if outputlog.stat().st_size > 0:
            fh.write("\n")
        fh.write(payload)
    return len(unique_events)


def ingest_transcripts(
    state: Dict[str, Any],
    projects_root: Path,
    codex_sessions_dir: Path,
    claude_projects_dir: Path,
    project_filter: Optional[set[Path]],
    initial_tail_bytes: int,
    first_seen_history: str,
    reindex: bool,
    dry_run: bool,
    verbose: bool,
) -> Dict[str, Any]:
    events_by_project: Dict[str, List[OutputEvent]] = {}
    source_errors: List[Dict[str, str]] = []

    if project_filter is not None and len(project_filter) == 0:
        return {"project_event_counts": {}, "projects_touched": [], "note": "no_target_projects"}

    codex_files = sorted(codex_sessions_dir.glob("**/*.jsonl")) if codex_sessions_dir.is_dir() else []
    claude_files: List[Path] = []
    if claude_projects_dir.is_dir():
        for path in sorted(claude_projects_dir.glob("**/*.jsonl")):
            parts = {p.lower() for p in path.parts}
            if "subagents" in parts:
                continue
            claude_files.append(path)

    if verbose:
        print(f"Scanning codex files: {len(codex_files)}")
        print(f"Scanning claude files: {len(claude_files)}")

    for source_path in codex_files:
        key = str(source_path)
        prior_entry = state["sources"].get(key)
        prior_entry_snapshot = dict(prior_entry) if isinstance(prior_entry, dict) else prior_entry
        try:
            existing = state["sources"].get(key, {})
            if project_filter is not None and project_filter:
                hinted_project = hint_source_project(
                    "codex",
                    source_path,
                    existing if isinstance(existing, dict) else {},
                    projects_root,
                )
                if hinted_project is None:
                    # Unknown-hint source: parse speculatively without mutating global state.
                    tmp_state = {"sources": {key: dict(existing) if isinstance(existing, dict) else {}}}
                    tmp_entry = ensure_source_state(
                        tmp_state,
                        source_path,
                        "codex",
                        initial_tail_bytes,
                        first_seen_history,
                        reindex,
                    )
                    events, _ = parse_codex_events(source_path, tmp_entry, projects_root, project_filter)
                    if events:
                        state["sources"][key] = tmp_entry
                        update_session_tracking(state, "codex", source_path, tmp_entry, events, projects_root)
                        for evt in events:
                            events_by_project.setdefault(str(evt.project_dir), []).append(evt)
                    continue
                if hinted_project not in project_filter:
                    continue

            entry = ensure_source_state(
                state,
                source_path,
                "codex",
                initial_tail_bytes,
                first_seen_history,
                reindex,
            )
            events, _ = parse_codex_events(source_path, entry, projects_root, project_filter)
            update_session_tracking(state, "codex", source_path, entry, events, projects_root)
            for evt in events:
                events_by_project.setdefault(str(evt.project_dir), []).append(evt)
        except Exception as exc:
            if prior_entry_snapshot is None:
                state["sources"].pop(key, None)
            else:
                state["sources"][key] = prior_entry_snapshot
            source_errors.append({"source": "codex", "path": str(source_path), "error": str(exc)})
            if verbose:
                print(f"[warn] codex source parse failed: {source_path}: {exc}", file=sys.stderr)

    for source_path in claude_files:
        key = str(source_path)
        prior_entry = state["sources"].get(key)
        prior_entry_snapshot = dict(prior_entry) if isinstance(prior_entry, dict) else prior_entry
        try:
            existing = state["sources"].get(key, {})
            if project_filter is not None and project_filter:
                hinted_project = hint_source_project(
                    "claude",
                    source_path,
                    existing if isinstance(existing, dict) else {},
                    projects_root,
                )
                if hinted_project is None:
                    tmp_state = {"sources": {key: dict(existing) if isinstance(existing, dict) else {}}}
                    tmp_entry = ensure_source_state(
                        tmp_state,
                        source_path,
                        "claude",
                        initial_tail_bytes,
                        first_seen_history,
                        reindex,
                    )
                    events, _ = parse_claude_events(source_path, tmp_entry, projects_root, project_filter)
                    if events:
                        state["sources"][key] = tmp_entry
                        update_session_tracking(state, "claude", source_path, tmp_entry, events, projects_root)
                        for evt in events:
                            events_by_project.setdefault(str(evt.project_dir), []).append(evt)
                    continue
                if hinted_project not in project_filter:
                    continue

            entry = ensure_source_state(
                state,
                source_path,
                "claude",
                initial_tail_bytes,
                first_seen_history,
                reindex,
            )
            events, _ = parse_claude_events(source_path, entry, projects_root, project_filter)
            update_session_tracking(state, "claude", source_path, entry, events, projects_root)
            for evt in events:
                events_by_project.setdefault(str(evt.project_dir), []).append(evt)
        except Exception as exc:
            if prior_entry_snapshot is None:
                state["sources"].pop(key, None)
            else:
                state["sources"][key] = prior_entry_snapshot
            source_errors.append({"source": "claude", "path": str(source_path), "error": str(exc)})
            if verbose:
                print(f"[warn] claude source parse failed: {source_path}: {exc}", file=sys.stderr)

    appended_counts: Dict[str, int] = {}
    for project_key, events in events_by_project.items():
        project_dir = Path(project_key)
        events.sort(key=lambda e: e.timestamp)
        appended = append_output_events(project_dir, events, dry_run=dry_run)
        appended_counts[project_key] = appended

    result: Dict[str, Any] = {
        "project_event_counts": appended_counts,
        "projects_touched": sorted(events_by_project.keys()),
    }
    sessions = state.get("sessions")
    if isinstance(sessions, dict):
        result["session_tracking_count"] = len(sessions)
        result["sessions_ingested_count"] = sum(1 for v in sessions.values() if isinstance(v, dict) and v.get("ingested"))
    if source_errors:
        result["source_errors"] = source_errors
    return result


def read_unread_after_marker(outputlog_path: Path) -> str:
    if not outputlog_path.exists():
        return ""
    text = outputlog_path.read_text(encoding="utf-8")
    marker_index = text.rfind(MARKER_LINE)
    if marker_index < 0:
        # Recovery behavior: if marker was manually removed, treat entire file as unread.
        return text.strip()
    unread = text[marker_index + len(MARKER_LINE) :]
    return unread.strip()


def read_outputlog_without_marker(outputlog_path: Path) -> str:
    if not outputlog_path.exists():
        return ""
    text = outputlog_path.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if line.strip() != MARKER_LINE]
    return "\n".join(lines).strip()


def move_marker_to_eof(outputlog_path: Path, dry_run: bool = False) -> None:
    text = outputlog_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    lines = [line for line in lines if line.strip() != MARKER_LINE]
    rebuilt = "\n".join(lines).rstrip() + "\n\n" + MARKER_LINE + "\n"
    if dry_run:
        return
    outputlog_path.write_text(rebuilt, encoding="utf-8")


def ensure_summarylog(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(SUMMARYLOG_HEADER, encoding="utf-8")


def append_summary_entry(path: Path, summary_entry_md: str, dry_run: bool = False) -> None:
    ensure_summarylog(path)
    entry = summary_entry_md.rstrip() + "\n\n"
    if dry_run:
        return
    with path.open("a", encoding="utf-8") as fh:
        fh.write(entry)


def read_status_file(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return DEFAULT_STATUS


def trim_text_block(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    return head + "\n\n[TRUNCATED]\n"


def build_summary_prompt(project_dir: Path, unread_text: str, current_status: str, max_unread_chars: int) -> str:
    unread_block = trim_text_block(unread_text, max_unread_chars)
    return f"""
You are updating local project tracking files.

Project path: {project_dir}
Current UTC time: {now_utc_iso()}

Inputs:
1) Unread assistant outputs from outputlog marker -> EOF.
2) Existing project status dashboard content.

Write a JSON object with keys:
- summary_entry_md: markdown to append to .agent/summarylog.md. Include a UTC timestamp heading and concise, evidence-based bullets.
- projectstatus_md: full replacement markdown for .agent/projectstatus.md.

Requirements for projectstatus_md:
- Human-first, agent-second context.
- Information dense but easy to scan.
- Explicitly include sections for:
  - Active Threads
  - Subthreads
  - Ideas Not Started
  - Ideas In Progress
  - Recently Finished
  - Absolute Latest Event
- If something is unknown, say "Unknown" instead of inventing details.

Output requirements:
- Return JSON only.
- Do not include markdown fences.
- Do not include explanatory text before or after JSON.

Unread output chunk:
-----BEGIN_UNREAD-----
{unread_block}
-----END_UNREAD-----

Existing project status:
-----BEGIN_CURRENT_STATUS-----
{current_status}
-----END_CURRENT_STATUS-----
""".strip()


def extract_json_object(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    if not text:
        raise ValueError("empty LLM output")

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1)
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

    decoder = json.JSONDecoder()
    start = text.find("{")
    while start >= 0:
        try:
            obj, _ = decoder.raw_decode(text[start:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        start = text.find("{", start + 1)

    raise ValueError("could not parse JSON object from LLM output")


def parse_summary_payload(raw: str) -> Tuple[str, str]:
    payload = extract_json_object(raw)
    summary_entry = str(payload.get("summary_entry_md", "")).strip()
    projectstatus_md = str(payload.get("projectstatus_md", "")).strip()
    if not summary_entry or not projectstatus_md:
        raise ValueError("missing summary_entry_md or projectstatus_md")
    return summary_entry + "\n", projectstatus_md + "\n"


def claude_summarize(
    project_dir: Path,
    unread_text: str,
    current_status: str,
    max_unread_chars: int,
    claude_settings: str,
    claude_betas: str,
    claude_skip_permissions: bool,
) -> Tuple[str, str]:
    prompt = build_summary_prompt(project_dir, unread_text, current_status, max_unread_chars)
    cmd = ["claude"]
    if claude_settings:
        cmd.extend(["--settings", claude_settings])
    if claude_skip_permissions:
        cmd.append("--dangerously-skip-permissions")
    if claude_betas:
        cmd.extend(["--betas", claude_betas])
    cmd.extend(["-p", prompt])
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=LLM_TIMEOUT_SECONDS)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        raise RuntimeError(f"claude summarize failed (code {proc.returncode}). stderr={stderr} stdout={stdout}")
    raw = (proc.stdout or "").strip()
    return parse_summary_payload(raw)


def codex_summarize(
    project_dir: Path,
    unread_text: str,
    current_status: str,
    max_unread_chars: int,
) -> Tuple[str, str]:
    prompt = build_summary_prompt(project_dir, unread_text, current_status, max_unread_chars)

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["summary_entry_md", "projectstatus_md"],
        "properties": {
            "summary_entry_md": {"type": "string", "minLength": 1},
            "projectstatus_md": {"type": "string", "minLength": 1},
        },
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as schema_file:
        json.dump(schema, schema_file)
        schema_path = schema_file.name

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as out_file:
        out_path = out_file.name

    cmd = [
        "codex",
        "exec",
        "--color",
        "never",
        "-C",
        str(project_dir),
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--output-schema",
        schema_path,
        "--output-last-message",
        out_path,
        "-",
    ]

    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=LLM_TIMEOUT_SECONDS,
        )
    finally:
        # Keep schema/output files until parse step; remove in final block.
        pass

    try:
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            raise RuntimeError(f"codex exec failed (code {proc.returncode}). stderr={stderr} stdout={stdout}")

        raw = Path(out_path).read_text(encoding="utf-8").strip()
        return parse_summary_payload(raw)
    finally:
        for tmp in (schema_path, out_path):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def fallback_summarize(project_dir: Path, unread_text: str) -> Tuple[str, str]:
    lines = [line.strip() for line in unread_text.splitlines() if line.strip()]
    interesting = [line for line in lines if not line.startswith("### [") and line != MARKER_LINE]
    bullets = interesting[:8]
    if not bullets:
        bullets = ["No substantial new text in unread window."]

    timestamp = now_utc_iso()
    summary_entry = [f"## {timestamp} UTC", "", "- Fallback summary used (configured LLM unavailable)."]
    summary_entry.extend(f"- {b}" for b in bullets)

    latest = bullets[0] if bullets else "Unknown"
    projectstatus = f"""# Project Status Dashboard

Last updated: {timestamp}

## Absolute Latest Event

- {latest}

## Active Threads

- Unknown

## Subthreads

- Unknown

## Ideas Not Started

- Unknown

## Ideas In Progress

- Unknown

## Recently Finished

- Unknown
"""
    return "\n".join(summary_entry) + "\n", projectstatus


def summarize_project(
    project_dir: Path,
    engine: str,
    max_unread_chars: int,
    claude_settings: str,
    claude_betas: str,
    claude_skip_permissions: bool,
    dry_run: bool,
    verbose: bool,
) -> Dict[str, Any]:
    outputlog = project_dir / ".agent" / "outputlog.md"
    summarylog = project_dir / ".agent" / "summarylog.md"
    projectstatus = project_dir / ".agent" / "projectstatus.md"

    unread_text = read_unread_after_marker(outputlog)
    bootstrap_from_history = False
    if not unread_text:
        if projectstatus.exists() and projectstatus.read_text(encoding="utf-8").strip():
            return {"project": str(project_dir), "processed": False, "reason": "no_unread_content"}
        history_text = read_outputlog_without_marker(outputlog)
        if not history_text:
            if not dry_run:
                projectstatus.parent.mkdir(parents=True, exist_ok=True)
                projectstatus.write_text(DEFAULT_STATUS, encoding="utf-8")
            return {"project": str(project_dir), "processed": False, "reason": "no_unread_content"}
        unread_text = history_text
        bootstrap_from_history = True
    if dry_run:
        return {
            "project": str(project_dir),
            "processed": False,
            "reason": "dry_run_skipped_summary",
            "unread_chars": len(unread_text),
            "bootstrap_from_history": bootstrap_from_history,
        }

    current_status = read_status_file(projectstatus)

    try:
        if engine == "claude":
            summary_entry_md, projectstatus_md = claude_summarize(
                project_dir=project_dir,
                unread_text=unread_text,
                current_status=current_status,
                max_unread_chars=max_unread_chars,
                claude_settings=claude_settings,
                claude_betas=claude_betas,
                claude_skip_permissions=claude_skip_permissions,
            )
        elif engine == "codex":
            summary_entry_md, projectstatus_md = codex_summarize(
                project_dir=project_dir,
                unread_text=unread_text,
                current_status=current_status,
                max_unread_chars=max_unread_chars,
            )
        else:
            raise RuntimeError(f"Unsupported engine: {engine}")
    except Exception as exc:
        if verbose:
            print(f"[{project_dir}] summarization fallback: {exc}", file=sys.stderr)
        summary_entry_md, projectstatus_md = fallback_summarize(project_dir, unread_text)

    append_summary_entry(summarylog, summary_entry_md, dry_run=dry_run)
    if not dry_run:
        projectstatus.parent.mkdir(parents=True, exist_ok=True)
        projectstatus.write_text(projectstatus_md, encoding="utf-8")

    move_marker_to_eof(outputlog, dry_run=dry_run)

    return {
        "project": str(project_dir),
        "processed": True,
        "summary_path": str(summarylog),
        "status_path": str(projectstatus),
        "bootstrap_from_history": bootstrap_from_history,
    }


def resolve_target_projects(
    args: argparse.Namespace,
    projects_root: Path,
) -> List[Path]:
    explicit = [Path(p).resolve() for p in args.project]
    if explicit:
        return sorted(explicit, key=lambda p: str(p))

    if args.all_projects:
        return []  # Empty list means no active filter.

    return discover_active_projects(projects_root, verbose=args.verbose)


def run_cycle(args: argparse.Namespace) -> int:
    projects_root = Path(args.projects_root)
    state_file = Path(args.state_file)
    codex_sessions_dir = Path(args.codex_sessions_dir)
    claude_projects_dir = Path(args.claude_projects_dir)

    state = load_state(state_file)
    target_projects = resolve_target_projects(args, projects_root)

    if args.verbose:
        if target_projects:
            print("Target projects:")
            for p in target_projects:
                print(f"- {p}")
        elif args.all_projects:
            print("Target projects: all projects seen in transcript events")
        else:
            print("No active tmux projects detected; nothing to process unless --all-projects or --project is used.")

    if not target_projects and not args.all_projects:
        print(json.dumps({"ingest": {"project_event_counts": {}, "projects_touched": [], "note": "no_target_projects"}, "summaries": []}, indent=2))
        return 0

    if target_projects:
        project_filter = set(target_projects)
    elif args.all_projects:
        project_filter = None
    else:
        project_filter = set()

    ingest_result = ingest_transcripts(
        state=state,
        projects_root=projects_root,
        codex_sessions_dir=codex_sessions_dir,
        claude_projects_dir=claude_projects_dir,
        project_filter=project_filter,
        initial_tail_bytes=args.initial_tail_bytes,
        first_seen_history=args.first_seen_history,
        reindex=args.reindex,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    if not args.dry_run:
        save_state(state_file, state)

    touched_projects = [Path(p) for p in ingest_result.get("projects_touched", [])]
    if target_projects:
        summarize_targets = target_projects
    elif args.all_projects:
        summarize_targets_map: Dict[str, Path] = {str(p): p for p in touched_projects}
        if projects_root.is_dir():
            for project_dir in projects_root.iterdir():
                if (project_dir / ".agent" / "outputlog.md").exists():
                    summarize_targets_map[str(project_dir)] = project_dir
        summarize_targets = sorted(summarize_targets_map.values(), key=lambda p: str(p))
    else:
        summarize_targets = []

    summary_results: List[Dict[str, Any]] = []
    for project_dir in summarize_targets:
        if not project_dir.exists():
            continue
        result = summarize_project(
            project_dir=project_dir,
            engine=args.engine,
            max_unread_chars=args.max_unread_chars,
            claude_settings=args.claude_settings,
            claude_betas=args.claude_betas,
            claude_skip_permissions=args.claude_skip_permissions,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        summary_results.append(result)

    print(json.dumps({"ingest": ingest_result, "summaries": summary_results}, indent=2))
    return 0


def run_ingest_only(args: argparse.Namespace) -> int:
    projects_root = Path(args.projects_root)
    state_file = Path(args.state_file)
    codex_sessions_dir = Path(args.codex_sessions_dir)
    claude_projects_dir = Path(args.claude_projects_dir)
    state = load_state(state_file)

    target_projects = resolve_target_projects(args, projects_root)
    if not target_projects and not args.all_projects:
        print(json.dumps({"project_event_counts": {}, "projects_touched": [], "note": "no_target_projects"}, indent=2))
        return 0

    if target_projects:
        project_filter = set(target_projects)
    elif args.all_projects:
        project_filter = None
    else:
        project_filter = set()

    result = ingest_transcripts(
        state=state,
        projects_root=projects_root,
        codex_sessions_dir=codex_sessions_dir,
        claude_projects_dir=claude_projects_dir,
        project_filter=project_filter,
        initial_tail_bytes=args.initial_tail_bytes,
        first_seen_history=args.first_seen_history,
        reindex=args.reindex,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    if not args.dry_run:
        save_state(state_file, state)

    print(json.dumps(result, indent=2))
    return 0


def run_summarize_only(args: argparse.Namespace) -> int:
    projects_root = Path(args.projects_root)
    target_projects = resolve_target_projects(args, projects_root)

    if not target_projects and not args.all_projects:
        print(json.dumps({"summaries": [], "note": "no_target_projects"}, indent=2))
        return 0

    if args.all_projects:
        # In summarize-only mode with --all-projects, scan projects root for .agent/outputlog.md.
        if not projects_root.is_dir():
            print(json.dumps({"summaries": [], "note": "projects_root_missing"}, indent=2))
            return 0
        target_projects = sorted(
            [p for p in projects_root.iterdir() if (p / ".agent" / "outputlog.md").exists()],
            key=lambda p: str(p),
        )

    results = []
    for project_dir in target_projects:
        result = summarize_project(
            project_dir=project_dir,
            engine=args.engine,
            max_unread_chars=args.max_unread_chars,
            claude_settings=args.claude_settings,
            claude_betas=args.claude_betas,
            claude_skip_permissions=args.claude_skip_permissions,
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        results.append(result)

    print(json.dumps({"summaries": results}, indent=2))
    return 0


def run_discover_only(args: argparse.Namespace) -> int:
    projects_root = Path(args.projects_root)
    projects = discover_active_projects(projects_root, verbose=args.verbose)
    print(json.dumps({"active_projects": [str(p) for p in projects]}, indent=2))
    return 0


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    command = args.command

    if command == "run":
        return run_cycle(args)
    if command == "discover":
        return run_discover_only(args)
    if command == "ingest":
        return run_ingest_only(args)
    if command == "summarize":
        return run_summarize_only(args)

    raise RuntimeError(f"Unknown command: {command}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
