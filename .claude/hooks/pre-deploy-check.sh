#!/usr/bin/env bash
# pre-deploy-check.sh
# Trigger: PreToolUse Bash
# Bloqueia deploy se typecheck ou testes mínimos falharem.

set -euo pipefail

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tool_input',{}).get('command',''))" 2>/dev/null || echo "")

# Só age em comandos de deploy
if ! echo "$COMMAND" | grep -qiE 'deploy|publish|release|npm run (build|start)|docker (build|push|compose up)'; then
  exit 0
fi

echo "[pre-deploy-check] Comando de deploy detectado. Rodando verificações obrigatórias..." >&2

ERRORS=0

# 1. Typecheck (placeholder — adaptar quando tsconfig existir)
if [ -f "tsconfig.json" ]; then
  echo "[pre-deploy-check] Rodando tsc --noEmit..." >&2
  if ! npx tsc --noEmit 2>/dev/null; then
    echo "[pre-deploy-check] ERRO: TypeScript typecheck falhou." >&2
    ERRORS=$((ERRORS + 1))
  else
    echo "[pre-deploy-check] OK: typecheck passou." >&2
  fi
else
  echo "[pre-deploy-check] AVISO: tsconfig.json não encontrado — typecheck ignorado." >&2
fi

# 2. Smoke test (placeholder — adaptar quando fixtures existirem)
if [ -f "package.json" ] && grep -q '"test:smoke"' package.json 2>/dev/null; then
  echo "[pre-deploy-check] Rodando npm run test:smoke..." >&2
  if ! npm run test:smoke 2>/dev/null; then
    echo "[pre-deploy-check] ERRO: Smoke test falhou." >&2
    ERRORS=$((ERRORS + 1))
  else
    echo "[pre-deploy-check] OK: smoke test passou." >&2
  fi
else
  echo "[pre-deploy-check] AVISO: test:smoke não configurado — teste ignorado." >&2
fi

# 3. Validação de classes do modelo ativo (placeholder)
if [ -f "package.json" ] && grep -q '"validate:model"' package.json 2>/dev/null; then
  echo "[pre-deploy-check] Rodando npm run validate:model..." >&2
  if ! npm run validate:model 2>/dev/null; then
    echo "[pre-deploy-check] ERRO: Validação do modelo falhou." >&2
    ERRORS=$((ERRORS + 1))
  fi
fi

if [ "$ERRORS" -gt 0 ]; then
  echo "[pre-deploy-check] BLOQUEADO: $ERRORS verificação(ões) falharam. Corrija antes de fazer deploy." >&2
  exit 1
fi

echo "[pre-deploy-check] Todas as verificações passaram. Deploy liberado." >&2
exit 0
