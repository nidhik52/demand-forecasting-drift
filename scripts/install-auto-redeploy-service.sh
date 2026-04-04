#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="retail-auto-redeploy.service"
SOURCE_FILE="$(cd "$(dirname "$0")" && pwd)/$SERVICE_NAME"
TARGET_FILE="/etc/systemd/system/$SERVICE_NAME"

if [[ ! -f "$SOURCE_FILE" ]]; then
  echo "Service definition not found: $SOURCE_FILE"
  exit 1
fi

sudo cp "$SOURCE_FILE" "$TARGET_FILE"
sudo systemctl daemon-reload
sudo systemctl enable --now "$SERVICE_NAME"
sudo systemctl status "$SERVICE_NAME" --no-pager