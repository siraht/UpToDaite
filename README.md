# Project Tracker

This tool tracks assistant outputs for active projects under `/data/projects/*` and keeps three files up to date in each project:

- `.agent/outputlog.md`: raw assistant outputs (Codex + Claude).
- `.agent/summarylog.md`: periodic summary entries.
- `.agent/projectstatus.md`: current dashboard/control-surface status.

## Marker Behavior

`outputlog.md` uses this marker line:

`<!-- AGENT_TRACKER_MARKER -->`

Unread content is everything after the marker to EOF.

- Ingestion appends new entries at EOF (after marker).
- Summarization reads marker -> EOF.
- On success, marker is moved to EOF so the unread window is reset.

## Commands

Run full pipeline:

```bash
python3 tools/project_tracker/tracker.py run --verbose
```

Default summarization engine is `claude`. Use `--engine codex` to switch.

Dry run:

```bash
python3 tools/project_tracker/tracker.py run --dry-run --verbose
```

`--dry-run` does not call the LLM summarizer and does not mutate files.

Process one project:

```bash
python3 tools/project_tracker/tracker.py run --project /data/projects/research
```

## Cron (Every 30 Minutes)

Install/update cron entry:

```bash
bash tools/project_tracker/install_cron.sh
```

This creates an idempotent entry tagged with `# project-tracker-30m` and logs to `~/.agent-tracker/cron.log`.

## State File

Per-source offsets are kept in `~/.agent-tracker/state.json`.
