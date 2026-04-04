#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PRUNE_SCRIPT="$SCRIPT_DIR/docker-weekly-prune.sh"
CRON_EXPR="0 3 * * 0"
CRON_CMD="$PRUNE_SCRIPT >/tmp/docker-weekly-prune.log 2>&1"
CRON_LINE="$CRON_EXPR $CRON_CMD"

chmod +x "$PRUNE_SCRIPT"

CURRENT_CRON="$(crontab -l 2>/dev/null || true)"

if printf '%s\n' "$CURRENT_CRON" | grep -Fq "$PRUNE_SCRIPT"; then
  echo "Weekly Docker prune cron already exists."
  exit 0
fi

{
  printf '%s\n' "$CURRENT_CRON"
  printf '%s\n' "$CRON_LINE"
} | crontab -

echo "Installed weekly Docker prune cron: $CRON_EXPR"