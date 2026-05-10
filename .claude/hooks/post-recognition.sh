#!/usr/bin/env bash
# post-recognition.sh
# Trigger: PostToolUse Bash
# Salva telemetria após execução do pipeline de reconhecimento.
# Não bloqueia — apenas registra.

set -euo pipefail

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('command',''))" 2>/dev/null || echo "")
EXIT_CODE=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_response',{}).get('exit_code', 'unknown'))" 2>/dev/null || echo "unknown")

# Só registra se o comando for do pipeline de reconhecimento
if ! echo "$COMMAND" | grep -qiE 'orchestrator|yolo|dinov2|inference|detect|recognize|pipeline'; then
  exit 0
fi

TELEMETRY_DIR="/tmp/quantx-telemetry"
mkdir -p "$TELEMETRY_DIR"

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
LOG_FILE="$TELEMETRY_DIR/$(date +%Y%m%d)_recognition.jsonl"

# Registrar evento de telemetria
python3 - <<PYEOF >> "$LOG_FILE" 2>/dev/null || true
import json, os, datetime

event = {
    "timestamp": "$TIMESTAMP",
    "event": "recognition_pipeline_executed",
    "command_snippet": "$COMMAND"[:120].replace('"', "'"),
    "exit_code": "$EXIT_CODE",
    "tenant_id": os.environ.get("QUANTX_TENANT_ID", "unknown"),
    "plan_id": os.environ.get("QUANTX_PLAN_ID", "unknown"),
    "session_id": os.environ.get("CLAUDE_SESSION_ID", "unknown")
}
print(json.dumps(event))
PYEOF

echo "[post-recognition] Telemetria registrada em $LOG_FILE." >&2
exit 0
