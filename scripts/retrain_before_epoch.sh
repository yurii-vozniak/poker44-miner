#!/usr/bin/env bash
# Retrain and restart the miner shortly before each competition epoch boundary.
# Competition epochs start at 12:00 UTC every 132h (see api.poker44.net competition/current).
set -euo pipefail

REPO_DIR="/root/workspaces/projects/poker44"
STATE_FILE="${REPO_DIR}/data/benchmark/last_epoch_prep.txt"
LOG_FILE="${REPO_DIR}/data/benchmark/epoch_retrain.log"
ENV_FILE="${REPO_DIR}/.env"
API_URL="${POKER44_COMPETITION_API:-https://api.poker44.net/api/v1/competition/current}"
PREP_LEAD_SECONDS="${POKER44_EPOCH_PREP_LEAD_SECONDS:-7200}"
POST_START_GRACE_SECONDS="${POKER44_EPOCH_PREP_GRACE_SECONDS:-1800}"
TRAIN_DATES="${POKER44_TRAIN_DATES:-30}"
HOLDOUT_DATES="${POKER44_HOLDOUT_DATES:-5}"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY || true

mkdir -p "${REPO_DIR}/data/benchmark"
# shellcheck disable=SC1091
source "${REPO_DIR}/miner_env/bin/activate"
export PYTHONPATH="${REPO_DIR}"
cd "${REPO_DIR}"

log() {
  local message="[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*"
  echo "${message}"
  echo "${message}" >> "${LOG_FILE}"
}

read_state() {
  if [ -f "${STATE_FILE}" ]; then
    tr -d '[:space:]' < "${STATE_FILE}"
  fi
}

write_state() {
  printf '%s' "$1" > "${STATE_FILE}"
}

eval "$(python - <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

api_url = os.environ.get("POKER44_COMPETITION_API", "https://api.poker44.net/api/v1/competition/current")
prep_lead = int(os.environ.get("POKER44_EPOCH_PREP_LEAD_SECONDS", "7200"))
grace = int(os.environ.get("POKER44_EPOCH_PREP_GRACE_SECONDS", "1800"))

try:
    with urllib.request.urlopen(api_url, timeout=30) as response:
        payload = json.load(response)
except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
    print(f'echo "Failed to fetch competition status: {exc}" >&2', file=sys.stdout)
    print("exit 1", file=sys.stdout)
    sys.exit(0)

data = payload.get("data") or {}
epoch = data.get("epoch") or {}
ends_at_raw = epoch.get("endsAt")
starts_at_raw = epoch.get("startsAt")
current_epoch_id = str(epoch.get("epochId") or "")

if not ends_at_raw:
    print('echo "Competition API returned no epoch.endsAt" >&2', file=sys.stdout)
    print("exit 1", file=sys.stdout)
    sys.exit(0)

ends_at = datetime.fromisoformat(ends_at_raw.replace("Z", "+00:00"))
now = datetime.now(timezone.utc)
seconds_remaining = (ends_at - now).total_seconds()
next_epoch_id = f"day_{ends_at.date().isoformat()}_1200utc"

in_window = (-grace) <= seconds_remaining <= prep_lead
should_retrain = "1" if in_window else "0"

print(f'CURRENT_EPOCH_ID="{current_epoch_id}"')
print(f'NEXT_EPOCH_ID="{next_epoch_id}"')
print(f'EPOCH_ENDS_AT="{ends_at_raw}"')
print(f'SECONDS_REMAINING="{int(seconds_remaining)}"')
print(f'SHOULD_RETRAIN="{should_retrain}"')
if starts_at_raw:
    print(f'EPOCH_STARTS_AT="{starts_at_raw}"')
PY
)"

if [ "${SHOULD_RETRAIN:-0}" != "1" ]; then
  log "Outside prep window for ${NEXT_EPOCH_ID} (seconds_remaining=${SECONDS_REMAINING})."
  exit 0
fi

last_prep="$(read_state)"
if [ "${last_prep}" = "${NEXT_EPOCH_ID}" ]; then
  log "Already prepped for ${NEXT_EPOCH_ID}; skipping."
  exit 0
fi

log "Pre-epoch retrain starting for ${NEXT_EPOCH_ID} (current=${CURRENT_EPOCH_ID}, seconds_remaining=${SECONDS_REMAINING})."

python deploy/download_benchmark.py --dates "${TRAIN_DATES}" --refresh
python deploy/train_stacked.py --dates "${TRAIN_DATES}" --holdout-dates "${HOLDOUT_DATES}"
bash scripts/poker44-miner validate
bash scripts/poker44-miner manifest-check

if [ -f "${ENV_FILE}" ] && command -v git >/dev/null 2>&1; then
  repo_commit="$(git -C "${REPO_DIR}" rev-parse HEAD 2>/dev/null || true)"
  if [ -n "${repo_commit}" ] && grep -q '^POKER44_MODEL_REPO_COMMIT=' "${ENV_FILE}"; then
    sed -i "s|^POKER44_MODEL_REPO_COMMIT=.*|POKER44_MODEL_REPO_COMMIT=${repo_commit}|" "${ENV_FILE}"
    log "Updated POKER44_MODEL_REPO_COMMIT=${repo_commit} in .env"
  fi
fi

bash scripts/poker44-miner restart
write_state "${NEXT_EPOCH_ID}"
log "Pre-epoch retrain complete for ${NEXT_EPOCH_ID}."
