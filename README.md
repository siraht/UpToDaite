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

Run full pipeline (discover active tmux projects, ingest, summarize):

```bash
python3 tools/project_tracker/tracker.py run --verbose
```

Default summarization engine is `claude`. Use `--engine codex` to switch.
Default first-seen ingestion mode is full history (`--first-seen-history full`), so new transcript files are read from byte 0 instead of tail-only.

When using Claude, tracker invokes the CLI with synthetic settings and interleaved thinking by default:

```bash
claude --settings ~/.claude/settings-synthetic.json --dangerously-skip-permissions --betas interleaved-thinking -p "<prompt>"
```

You can override with:

- `--claude-settings /path/to/settings.json`
- `--claude-betas value`
- `--no-claude-dangerously-skip-permissions`

Dry run:

```bash
python3 tools/project_tracker/tracker.py run --dry-run --verbose
```

`--dry-run` does not call the LLM summarizer; it reports unread sizes and planned actions without mutating files.

Discover active projects from tmux only:

```bash
python3 tools/project_tracker/tracker.py discover
```

Ingest only:

```bash
python3 tools/project_tracker/tracker.py ingest --verbose
```

Summarize only:

```bash
python3 tools/project_tracker/tracker.py summarize --verbose
```

Force Codex summarization:

```bash
python3 tools/project_tracker/tracker.py summarize --engine codex --verbose
```

Process a specific project (repeat `--project` as needed):

```bash
python3 tools/project_tracker/tracker.py run --project /data/projects/research
```

Ignore tmux filter and process all projects seen in transcript events:

```bash
python3 tools/project_tracker/tracker.py run --all-projects
```

## Cron (Every 30 Minutes)

Install/update cron entry:

```bash
bash tools/project_tracker/install_cron.sh
```

This creates an idempotent entry tagged with `# project-tracker-30m` and logs to:

`~/.agent-tracker/cron.log`

## State File

Per-source read offsets are kept in:

`~/.agent-tracker/state.json`

For unseen transcript files, the tracker starts according to `--first-seen-history`:

- `full` (default): starts from byte `0` (entire conversation history).
- `tail`: starts from a tail window (`--initial-tail-bytes`, default `200000`).

Use `--reindex` to rebuild source baselines with the current first-seen policy.

- Full-history baseline (default): `--first-seen-history full`
- Tail baseline: `--first-seen-history tail --initial-tail-bytes 200000`

To backfill all historical conversations for a project safely (dedupe is event-id based):

```bash
python3 tools/project_tracker/tracker.py run --project /data/projects/research --reindex --first-seen-history full --verbose
```

## Session Tracking

In addition to per-file offsets, tracker now keeps a session index in:

`~/.agent-tracker/state.json` under `sessions`.

Each session entry records source path, session id (when known), project hints, last offset scanned, event counts, and ingestion timestamps.

## Claude Filtering Notes

Claude transcript files contain many intermediate assistant records (`thinking`, planning text, and `tool_use` steps).  
Tracker keeps only assistant text tied to message IDs that do **not** include `tool_use`, and also drops short planning-prefixed text like `Let me check...` / `I'll start...` to avoid logging in-progress work chatter as final output.

## Rebuild Clean Logs

If older logs were ingested before filter improvements, rebuild a project from transcript history:

```bash
# optional: back up current logs first
cp /data/projects/<project>/.agent/outputlog.md /data/projects/<project>/.agent/outputlog.md.bak
cp /data/projects/<project>/.agent/summarylog.md /data/projects/<project>/.agent/summarylog.md.bak
cp /data/projects/<project>/.agent/projectstatus.md /data/projects/<project>/.agent/projectstatus.md.bak

# regenerate from full history with current filters
rm -f /data/projects/<project>/.agent/outputlog.md
rm -f /data/projects/<project>/.agent/summarylog.md
rm -f /data/projects/<project>/.agent/projectstatus.md
python3 tools/project_tracker/tracker.py run --project /data/projects/<project> --reindex --first-seen-history full --engine claude --verbose
```
