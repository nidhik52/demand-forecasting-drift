#!/usr/bin/env bash
set -euo pipefail

docker image prune -af
docker builder prune -af
docker container prune -f