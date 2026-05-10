#!/usr/bin/env bash
# post-human-review.sh
# Trigger: PostToolUse Edit
# Detecta quando uma correção humana foi salva e dispara Dataset Curation.
# Não bloqueia — apenas aciona o pipeline de aprendizado.

set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))" 2>/dev/null || echo "")

# Só age em arquivos de correção humana
if ! echo "$FILE_PATH" | grep -qiE 'correction|review|feedback|annotation'; then
  exit 0
fi

echo "[post-human-review] Correção humana detectada em: $FILE_PATH" >&2

TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
QUEUE_DIR="/tmp/quantx-curation-queue"
mkdir -p "$QUEUE_DIR"

# Extrair metadados básicos do arquivo de correção
CORRECTION_DATA=$(cat "$FILE_PATH" 2>/dev/null || echo "{}")

# Verificar campos obrigatórios (LGPD: reviewer_id obrigatório)
REVIEWER_ID=$(echo "$CORRECTION_DATA" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('reviewer_id',''))" 2>/dev/null || echo "")
CORRECTION_TIMESTAMP=$(echo "$CORRECTION_DATA" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('timestamp',''))" 2>/dev/null || echo "")

if [ -z "$REVIEWER_ID" ]; then
  echo "[post-human-review] ERRO: reviewer_id ausente — correção não pode ser processada (LGPD)." >&2
  echo "[post-human-review] A correção foi salva mas NÃO foi enfileirada para dataset curation." >&2
  exit 0
fi

if [ -z "$CORRECTION_TIMESTAMP" ]; then
  echo "[post-human-review] ERRO: timestamp ausente — correção não pode ser processada." >&2
  exit 0
fi

# Verificar que reviewer_id é hash anonimizado (não deve conter @ ou espaço)
if echo "$REVIEWER_ID" | grep -qE '@| |\.com|\.br'; then
  cat >&2 <<'EOF'
[post-human-review] ERRO LGPD: reviewer_id parece conter email ou nome real.
reviewer_id DEVE ser um hash anonimizado (ex: SHA256 do email).
A correção não foi enfileirada para proteger dados pessoais.
EOF
  exit 0
fi

# Enfileirar para Dataset Curation Agent
QUEUE_FILE="$QUEUE_DIR/$(date +%Y%m%d_%H%M%S%N)_correction.json"
python3 - <<PYEOF > "$QUEUE_FILE" 2>/dev/null || true
import json

event = {
    "queued_at": "$TIMESTAMP",
    "source_file": "$FILE_PATH",
    "reviewer_id": "$REVIEWER_ID",
    "correction_timestamp": "$CORRECTION_TIMESTAMP",
    "status": "pending_curation",
    "tenant_id": "$REVIEWER_ID"[:8]
}
print(json.dumps(event, indent=2))
PYEOF

echo "[post-human-review] Correção enfileirada para Dataset Curation: $QUEUE_FILE" >&2
echo "[post-human-review] Dataset Curation Agent processará em background (re-embed DINOv2 imediato)." >&2

# Contar fila para verificar threshold de fine-tune
QUEUE_SIZE=$(ls "$QUEUE_DIR"/*.json 2>/dev/null | wc -l || echo "0")
echo "[post-human-review] Fila atual: $QUEUE_SIZE correção(ões) pendentes." >&2

if [ "$QUEUE_SIZE" -ge 200 ]; then
  echo "[post-human-review] ALERTA: Threshold de fine-tune atingido ($QUEUE_SIZE amostras). Agendar YOLO fine-tune." >&2
fi

exit 0
