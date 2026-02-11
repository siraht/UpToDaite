#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRACKER_PY="${SCRIPT_DIR}/tracker.py"
LOG_PATH="${HOME}/.agent-tracker/cron.log"
STATE_PATH="${HOME}/.agent-tracker/state.json"

mkdir -p "${HOME}/.agent-tracker"
touch "${LOG_PATH}"

CRON_MARKER="# project-tracker-30m"
CRON_CMD="*/30 * * * * /usr/bin/env python3 '${TRACKER_PY}' run --state-file '${STATE_PATH}' >> '${LOG_PATH}' 2>&1 ${CRON_MARKER}"

EXISTING="$(crontab -l 2>/dev/null || true)"
FILTERED="$(printf '%s\n' "${EXISTING}" | sed "/${CRON_MARKER//\//\\/}/d")"

{
  printf '%s\n' "${FILTERED}"
  printf '%s\n' "${CRON_CMD}"
} | awk 'NF' | crontab -

echo "Installed cron entry:"
echo "${CRON_CMD}"
echo "Log file: ${LOG_PATH}"
