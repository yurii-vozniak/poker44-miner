#!/usr/bin/env bash
# Monitor v3 windows/scores and retrain when new v4.1 labeled data appears.
set -euo pipefail

REPO_DIR="/root/workspaces/projects/poker44"
LOG_FILE="${REPO_DIR}/data/benchmark/v3_monitor.log"
STATE_FILE="${REPO_DIR}/data/benchmark/v3_monitor_state.json"
MINER_UID="${POKER44_UID:-79}"
PM2_NAME="${PM2_NAME:-poker44_miner}"
RETRAIN_STAMP="${REPO_DIR}/data/benchmark/v3_last_retrain.txt"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy NO_PROXY no_proxy || true
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

should_retrain() {
  local v41_rows="$1"
  local prev_v41_rows="$2"
  if [ "${v41_rows}" -gt 0 ] && [ "${v41_rows}" != "${prev_v41_rows}" ]; then
    return 0
  fi
  if [ ! -f "${REPO_DIR}/models/micro_session_v2.joblib" ]; then
    return 0
  fi
  if [ ! -f "${RETRAIN_STAMP}" ]; then
    return 0
  fi
  local age_hours
  age_hours="$(( ( $(date +%s) - $(date -r "${RETRAIN_STAMP}" +%s) ) / 3600 ))"
  if [ "${age_hours}" -ge 24 ]; then
    return 0
  fi
  return 1
}

log "v3 monitor cycle start"

python deploy/v3_status.py --uid "${MINER_UID}" --json > "${REPO_DIR}/data/benchmark/v3_status_latest.json"

python - <<'PY' >> "${LOG_FILE}"
import json
from pathlib import Path

status = json.loads(Path("data/benchmark/v3_status_latest.json").read_text())
miner = status.get("miner") or {}
readiness = status.get("readiness") or {}
print(
    f"status uid={status.get('uid')} rank={status.get('rank')} "
    f"reward={float(miner.get('average_reward') or 0):.4f} "
    f"model={miner.get('model_version')} last={miner.get('last_evaluated_at')} "
    f"window={readiness.get('window_id')} active={readiness.get('collection_active')}"
)
PY

PREV_EVAL=""
PREV_V41_ROWS="0"
if [ -f "${STATE_FILE}" ]; then
  PREV_EVAL="$(python - <<'PY'
import json
from pathlib import Path
state = json.loads(Path("data/benchmark/v3_monitor_state.json").read_text())
print(state.get("last_evaluated_at", ""))
PY
)"
  PREV_V41_ROWS="$(python - <<'PY'
import json
from pathlib import Path
state = json.loads(Path("data/benchmark/v3_monitor_state.json").read_text())
print(state.get("v41_rows", 0))
PY
)"
fi

CUR_EVAL="$(python - <<'PY'
import json
from pathlib import Path
status = json.loads(Path("data/benchmark/v3_status_latest.json").read_text())
print((status.get("miner") or {}).get("last_evaluated_at", ""))
PY
)"

if [ -n "${CUR_EVAL}" ] && [ "${CUR_EVAL}" != "${PREV_EVAL}" ]; then
  log "NEW EVALUATION detected: ${PREV_EVAL} -> ${CUR_EVAL}"
fi

V41_ROWS="$(python - <<'PY'
from deploy.v41_benchmark_client import V41BenchmarkClient
from pathlib import Path
client = V41BenchmarkClient()
rows = client.download_jsonl(output_path=Path("data/micro_session_benchmark.jsonl"))
print(rows)
PY
)"

RETRAINED=0
if should_retrain "${V41_ROWS}" "${PREV_V41_ROWS}"; then
  if [ "${V41_ROWS}" -gt 0 ]; then
    log "Training micro-v2 on public v4.1 corpus (${V41_ROWS} rows)"
  else
    log "Daily proxy refresh for micro-v2 (no public v4.1 corpus yet)"
  fi
  python deploy/train_micro_session_v2.py --holdout-dates 5 --output models/micro_session_v2.joblib
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "${RETRAIN_STAMP}"
  RETRAINED=1
else
  log "Skipping retrain (v41_rows=${V41_ROWS}, last retrain recent)"
fi

python deploy/select_deploy_model.py --apply --holdout-dates 5 > "${REPO_DIR}/data/benchmark/v3_deploy_decision.json"
cp "${REPO_DIR}/data/benchmark/v3_deploy_decision.json" "${REPO_DIR}/data/benchmark/v3_deploy_state.json"

PREV_FACTORY=""
if [ -f "${STATE_FILE}" ]; then
  PREV_FACTORY="$(python - <<'PY'
import json
from pathlib import Path
state = json.loads(Path("data/benchmark/v3_monitor_state.json").read_text())
print(state.get("deploy_factory", ""))
PY
)"
fi

CUR_FACTORY="$(python - <<'PY'
import json
from pathlib import Path
print(json.loads(Path("data/benchmark/v3_deploy_decision.json").read_text()).get("factory",""))
PY
)"

if [ "${RETRAINED}" = "1" ] || [ "${CUR_FACTORY}" != "${PREV_FACTORY}" ]; then
  log "Restarting miner (factory=${CUR_FACTORY})"
  pm2 restart "${PM2_NAME}" --update-env || pm2 restart poker44_miner --update-env
fi

python - <<'PY' > "${STATE_FILE}"
import json
from pathlib import Path
status = json.loads(Path("data/benchmark/v3_status_latest.json").read_text())
decision = json.loads(Path("data/benchmark/v3_deploy_decision.json").read_text())
state = {
    "updated_at": status.get("checked_at"),
    "last_evaluated_at": (status.get("miner") or {}).get("last_evaluated_at"),
    "window_id": (status.get("readiness") or {}).get("window_id"),
    "collection_active": (status.get("readiness") or {}).get("collection_active"),
    "deploy_strategy": decision.get("strategy"),
    "deploy_factory": decision.get("factory"),
    "v41_rows": decision.get("metrics", {}).get("v41_rows", 0),
}
Path("data/benchmark/v3_monitor_state.json").write_text(json.dumps(state, indent=2) + "\n")
PY

log "v3 monitor cycle complete"
