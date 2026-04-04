#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspaces/demand-forecasting-drift}"
BRANCH="${BRANCH:-main}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-60}"

cd "$REPO_DIR"

if ! git rev-parse --verify "origin/$BRANCH" >/dev/null 2>&1; then
  echo "origin/$BRANCH not found. Run: git fetch origin $BRANCH"
  exit 1
fi

echo "Watching $REPO_DIR for remote changes on origin/$BRANCH..."

while true; do
  git fetch origin "$BRANCH" --quiet

  LOCAL_SHA="$(git rev-parse HEAD)"
  REMOTE_SHA="$(git rev-parse "origin/$BRANCH")"

  if [[ "$LOCAL_SHA" != "$REMOTE_SHA" ]]; then
    echo "Change detected at $(date -u +"%Y-%m-%dT%H:%M:%SZ"). Redeploying..."
    git pull --ff-only origin "$BRANCH"
    docker compose up -d --build
    docker image prune -af --filter "until=24h"
  fi

  read -r -t "$INTERVAL_SECONDS" _ || true
done