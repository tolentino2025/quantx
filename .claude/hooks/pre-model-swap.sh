#!/usr/bin/env bash
# pre-model-swap.sh
# Trigger: PreToolUse Edit
# Bloqueia troca de modelo YOLO ativo se .pt não existir ou classes divergirem do DB.

set -euo pipefail

INPUT=$(cat)
FILE_PATH=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('file_path',''))" 2>/dev/null || echo "")
NEW_CONTENT=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('new_string',''))" 2>/dev/null || echo "")

# Só age em arquivos que contenham model_path
if ! echo "$FILE_PATH $NEW_CONTENT" | grep -qiE 'model_path|model_version|active_model|\.pt'; then
  exit 0
fi

echo "[pre-model-swap] Mudança de modelo YOLO detectada." >&2
echo "[pre-model-swap] Arquivo: $FILE_PATH" >&2

# Extrair caminho do .pt do novo conteúdo
MODEL_PATH=$(echo "$NEW_CONTENT" | grep -oE '[a-zA-Z0-9_./-]+\.pt' | head -1 || echo "")

if [ -z "$MODEL_PATH" ]; then
  echo "[pre-model-swap] AVISO: Não foi possível extrair caminho do .pt — validação de arquivo ignorada." >&2
else
  echo "[pre-model-swap] Validando modelo: $MODEL_PATH" >&2

  # 1. Verificar se o arquivo .pt existe
  if [ ! -f "$MODEL_PATH" ]; then
    cat >&2 <<EOF
[pre-model-swap] BLOQUEADO: Arquivo do modelo não encontrado.
  Caminho: $MODEL_PATH

Verifique:
  - O arquivo foi copiado para o caminho correto?
  - O caminho está relativo ou absoluto corretamente?
  - O upload do modelo foi concluído?
EOF
    exit 1
  fi

  echo "[pre-model-swap] OK: Arquivo .pt encontrado em $MODEL_PATH." >&2

  # 2. Verificar classes do modelo vs DB (placeholder — adaptar quando yolo-registry-mcp existir)
  CLASSES_FILE="${MODEL_PATH%.pt}.classes.json"
  if [ -f "$CLASSES_FILE" ]; then
    echo "[pre-model-swap] Verificando classes do modelo contra DB..." >&2
    # Placeholder: comparação real requer acesso ao yolo-registry-mcp
    CLASS_COUNT=$(python3 -c "import json; d=json.load(open('$CLASSES_FILE')); print(len(d))" 2>/dev/null || echo "0")
    echo "[pre-model-swap] Modelo tem $CLASS_COUNT classes. Validação completa requer yolo-registry-mcp." >&2
  else
    echo "[pre-model-swap] AVISO: $CLASSES_FILE não encontrado — validação de classes ignorada." >&2
    echo "[pre-model-swap] AVISO: Certifique-se que as classes do modelo correspondem ao DB antes de ativar." >&2
  fi
fi

# 3. Exigir justificativa para troca de modelo
JUSTIFICATION="${QUANTX_MODEL_SWAP_JUSTIFICATION:-}"
if [ -z "$JUSTIFICATION" ]; then
  cat >&2 <<'EOF'
[pre-model-swap] BLOQUEADO: troca de modelo ativo exige justificativa.

Para prosseguir:
  export QUANTX_MODEL_SWAP_JUSTIFICATION="<motivo da troca>"

Requisitos:
  1. Versão anterior e nova versão do modelo
  2. Métricas de comparação (mAP, recall por classe)
  3. Plano de rollback se o novo modelo regredir
EOF
  exit 1
fi

echo "[pre-model-swap] Justificativa presente. Troca de modelo liberada." >&2
echo "[pre-model-swap] LEMBRETE: Após a troca, rode npm run validate:model para confirmar alinhamento com DB." >&2
exit 0
