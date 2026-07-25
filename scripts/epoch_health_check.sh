#!/usr/bin/env bash
# Quick health check before/during a competition epoch.
set -euo pipefail
REPO_DIR="/root/workspaces/projects/poker44"
cd "${REPO_DIR}"

echo "=== PM2 ==="
pm2 list 2>/dev/null || echo "PM2 not running"

echo "=== Axon port 8092 ==="
ss -tlnp 2>/dev/null | grep 8092 || echo "NOT LISTENING"

echo "=== Recent validator scores ==="
grep 'Scored' /root/.pm2/logs/poker44-miner-out.log 2>/dev/null | tail -3 || echo "No Scored logs yet"

echo "=== Model ==="
source miner_env/bin/activate
export PYTHONPATH=.
python - <<'PY'
import joblib
from pathlib import Path
p = Path("models/hybrid.joblib")
if p.is_file():
    a = joblib.load(p)
    m = a.get("metadata", {})
    print("hybrid.joblib OK | version", m.get("model_version"), "| rank_blend", a.get("rank_blend"))
else:
    print("hybrid.joblib MISSING")
PY

echo "=== Competition ==="
curl -s "https://api.poker44.net/api/v1/competition/current" | python3 -c "
import json,sys
d=json.load(sys.stdin).get('data',{})
e=d.get('epoch',{})
print('epoch', e.get('epochId'), 'ends', e.get('endsAt'), 'remaining_s', e.get('secondsRemaining'))
" 2>/dev/null || echo "API unreachable"
