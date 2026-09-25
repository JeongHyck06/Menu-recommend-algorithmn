#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
docker network create muckzi > /dev/null 2>&1 || true
docker compose up -d --build --wait --wait-timeout 600
docker image prune -f > /dev/null
