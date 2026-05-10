#!/usr/bin/env bash
# pre-threshold-change.sh
# Trigger: PreToolUse Edit
# Bloqueia edição de confidence-thresholds.json sem justificativa documentada.

set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))" 2>/dev/null || echo "")

# Só age em edições do arquivo de thresholds
if ! echo "$FILE_PATH" | grep -q "confidence-thresholds"; then
  exit 0
fi

echo "[pre-threshold-change] Edição de thresholds de confiança detectada." >&2
echo "[pre-threshold-change] Arquivo: $FILE_PATH" >&2

# Verificar se há justificativa no contexto da sessão (via variável de ambiente)
JUSTIFICATION="${QUANTX_THRESHOLD_JUSTIFICATION:-}"

if [ -z "$JUSTIFICATION" ]; then
  cat >&2 <<'EOF'
[pre-threshold-change] BLOQUEADO: mudança de threshold exige justificativa.

Para prosseguir, defina a variável de ambiente antes de editar:
  export QUANTX_THRESHOLD_JUSTIFICATION="<motivo da mudança>"

Requisitos para justificativa válida:
  1. Descrever qual classe e threshold está sendo alterado
  2. Informar a taxa de FP/FN atual que justifica a mudança
  3. Referenciar o relatório do False Positive Review Agent (se disponível)
  4. Estimar impacto: quantas detecções por semana serão afetadas

Exemplo:
  export QUANTX_THRESHOLD_JUSTIFICATION="Elevando auto_accept de extintor-pqs de 0.85 para 0.90. FP rate atual: 8.2% (acima do alerta de 8%). Relatório FP-2026-05-07 recomendou ajuste. Impacto estimado: +25 casos para verify_with_vision por semana."

A justificativa também deve constar no commit message.
EOF
  exit 1
fi

# Registrar log da mudança
LOG_DIR="/tmp/quantx-threshold-changes"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_threshold_change.log"

cat > "$LOG_FILE" <<EOF
Timestamp: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
File: $FILE_PATH
Justification: $JUSTIFICATION
EOF

echo "[pre-threshold-change] Justificativa registrada em $LOG_FILE." >&2
echo "[pre-threshold-change] Edição liberada. Lembre-se: a justificativa deve constar no commit message." >&2
exit 0
