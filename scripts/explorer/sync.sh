#!/bin/bash
# Push this repo to Explorer and pull the Gate B or live-track results back.
#
#   scripts/explorer/sync.sh push <nuid>
#   scripts/explorer/sync.sh pull <nuid>        # gate_b/
#   scripts/explorer/sync.sh pull-live <nuid>   # live/<season>/ archives and live/runs/
set -euo pipefail
MODE="${1:?push or pull}"
NUID="${2:?your NU username}"
HOST="login.explorer.northeastern.edu"
REMOTE="/scratch/$NUID/areaFluForecast"
LOCAL="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

case "$MODE" in
  push)
    rsync -avz --delete --exclude '.git/' --exclude '.venv/' --exclude '__pycache__/' --exclude '.DS_Store' \
      --exclude 'gate_b/' --exclude 'logs/' --exclude 'areaflu_runs/' \
      --exclude 'live/runs/' --exclude 'live/models/' --exclude 'live/[0-9][0-9][0-9][0-9]_[0-9][0-9]/' \
      "$LOCAL/" "$NUID@$HOST:$REMOTE/"
    ;;
  pull)
    rsync -avz "$NUID@$HOST:$REMOTE/gate_b/" "$LOCAL/gate_b/"
    ;;
  pull-live)
    # Never --delete here: archived forecasts are the prospective record.
    rsync -avz --include '[0-9][0-9][0-9][0-9]_[0-9][0-9]/***' --include 'runs/***' --include 'models/***' --exclude '*' \
      "$NUID@$HOST:$REMOTE/live/" "$LOCAL/live/"
    ;;
  *) echo "usage: sync.sh {push|pull|pull-live} <nuid>" >&2; exit 2 ;;
esac
