#!/usr/bin/env bash
# setup-runpod.sh — configura e sobe todos os serviços QuantX num RunPod sem Docker
# Uso: bash setup-runpod.sh
# Requer: RunPod com Ubuntu, acesso root
# Banco e Storage: Supabase (externo). Postgres local NUNCA é instalado.

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
ENV_FILE="$WORKSPACE/.env"
LOG_DIR="/tmp/quantx-logs"
mkdir -p "$LOG_DIR"

# ── cores ─────────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✔ $*${NC}"; }
info() { echo -e "${YELLOW}➜ $*${NC}"; }
fail() { echo -e "${RED}✘ $*${NC}"; exit 1; }

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║       QuantX — Setup RunPod (sem Docker)         ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ── 1. Variáveis de ambiente ──────────────────────────────────────────────────
info "Carregando variáveis de ambiente..."

# Carregar .env se existir
if [ -f "$ENV_FILE" ]; then
  set -a; source "$ENV_FILE"; set +a
  ok ".env carregado de $ENV_FILE"
fi

# Obrigatórias
SUPABASE_URL="${SUPABASE_URL:-}"
SUPABASE_SERVICE_ROLE_KEY="${SUPABASE_SERVICE_ROLE_KEY:-}"
DATABASE_URL="${DATABASE_URL:-}"

REDIS_PORT="${REDIS_PORT:-6379}"
ML_PORT="${ML_PORT:-8000}"
BACKEND_PORT="${BACKEND_PORT:-3000}"
MODEL_VERSION="${MODEL_VERSION:-yolov15-spci-2026.04}"
RENDER_DPI="${RENDER_DPI:-150}"

[ -n "$SUPABASE_URL" ]              || fail "SUPABASE_URL não definida — adicione ao $ENV_FILE"
[ -n "$SUPABASE_SERVICE_ROLE_KEY" ] || fail "SUPABASE_SERVICE_ROLE_KEY não definida — adicione ao $ENV_FILE"
[ -n "$DATABASE_URL" ]              || fail "DATABASE_URL não definida — adicione ao $ENV_FILE"

ok "Variáveis carregadas"

# ── 2. Dependências de sistema ────────────────────────────────────────────────
info "Verificando dependências do sistema..."

apt-get update -qq 2>/dev/null

PKGS=()
command -v redis-server &>/dev/null || PKGS+=(redis-server)
command -v pdftoppm     &>/dev/null || PKGS+=(poppler-utils)
command -v tesseract    &>/dev/null || PKGS+=(tesseract-ocr tesseract-ocr-por)
command -v curl         &>/dev/null || PKGS+=(curl)

if [ ${#PKGS[@]} -gt 0 ]; then
  info "Instalando: ${PKGS[*]}"
  apt-get install -y --no-install-recommends "${PKGS[@]}" -qq 2>/dev/null
fi

ok "Dependências OK"

# ── 3. Redis ──────────────────────────────────────────────────────────────────
info "Iniciando Redis..."
if redis-cli ping 2>/dev/null | grep -q PONG; then
  ok "Redis já está rodando"
else
  redis-server --daemonize yes \
    --logfile "$LOG_DIR/redis.log" \
    --port "$REDIS_PORT" \
    --save 900 1 --save 300 10
  sleep 1
  redis-cli ping | grep -q PONG && ok "Redis iniciado (porta $REDIS_PORT)" || fail "Redis não iniciou"
fi

# ── 4. Supabase — banco e storage ─────────────────────────────────────────────
info "Validando conexão com Supabase Postgres..."

# Testa conexão com o banco
if psql "$DATABASE_URL" -c "SELECT 1" > /dev/null 2>&1; then
  ok "Supabase Postgres acessível"
else
  fail "Não foi possível conectar ao Supabase Postgres — verifique DATABASE_URL"
fi

# Aplica migrations
MIGRATIONS_DIR="$WORKSPACE/quantx/backend/migrations"
if [ -d "$MIGRATIONS_DIR" ]; then
  info "Aplicando migrations..."
  for f in "$MIGRATIONS_DIR"/*.sql; do
    info "  $(basename $f)"
    psql "$DATABASE_URL" -f "$f" >> "$LOG_DIR/migrations.log" 2>&1 \
      && ok "  $(basename $f)" \
      || info "  $(basename $f) — verifique $LOG_DIR/migrations.log (pode já existir)"
  done
else
  info "Pasta de migrations não encontrada em $MIGRATIONS_DIR"
fi

info "Validando Supabase Storage..."

# Cria/valida buckets via REST API
for BUCKET in plans-original plans-pages plans-processed symbols-library training-datasets exports; do
  HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST "${SUPABASE_URL}/storage/v1/bucket" \
    -H "Authorization: Bearer ${SUPABASE_SERVICE_ROLE_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"id\":\"${BUCKET}\",\"name\":\"${BUCKET}\",\"public\":false}")
  if [ "$HTTP_STATUS" = "200" ] || [ "$HTTP_STATUS" = "201" ]; then
    ok "  bucket criado: $BUCKET"
  elif [ "$HTTP_STATUS" = "400" ] || [ "$HTTP_STATUS" = "409" ]; then
    ok "  bucket existe: $BUCKET"
  else
    info "  bucket $BUCKET: HTTP $HTTP_STATUS (verifique permissões do service_role_key)"
  fi
done

# Smoke test: upload, download e signed URL
info "Smoke test de storage..."
SMOKE_PATH="smoke-test/$(date +%s).txt"
SMOKE_DATA="quantx-storage-ok"

# Upload
HTTP_UP=$(curl -s -o /dev/null -w "%{http_code}" \
  -X PUT "${SUPABASE_URL}/storage/v1/object/plans-original/${SMOKE_PATH}" \
  -H "Authorization: Bearer ${SUPABASE_SERVICE_ROLE_KEY}" \
  -H "Content-Type: text/plain" \
  -H "x-upsert: true" \
  --data-raw "$SMOKE_DATA")
[ "$HTTP_UP" = "200" ] && ok "  upload OK" || fail "  upload falhou (HTTP $HTTP_UP)"

# Signed URL
HTTP_SIGN=$(curl -s -o /tmp/quantx-smoke-url.json -w "%{http_code}" \
  -X POST "${SUPABASE_URL}/storage/v1/object/sign/plans-original/${SMOKE_PATH}" \
  -H "Authorization: Bearer ${SUPABASE_SERVICE_ROLE_KEY}" \
  -H "Content-Type: application/json" \
  -d '{"expiresIn":60}')
[ "$HTTP_SIGN" = "200" ] && ok "  signed URL OK" || fail "  signed URL falhou (HTTP $HTTP_SIGN)"

SIGNED_URL=$(python3 -c "import json,sys; d=json.load(open('/tmp/quantx-smoke-url.json')); print('${SUPABASE_URL}' + d['signedURL'])" 2>/dev/null || echo "")
if [ -n "$SIGNED_URL" ]; then
  DOWN_DATA=$(curl -sf "$SIGNED_URL" 2>/dev/null || echo "")
  [ "$DOWN_DATA" = "$SMOKE_DATA" ] && ok "  download via signed URL OK" || info "  download retornou dados inesperados"
fi

ok "Supabase Storage validado"

# ── 5. ML Service (Python/FastAPI) ────────────────────────────────────────────
info "Iniciando ML service..."

pkill -f "uvicorn main:app" 2>/dev/null || true
sleep 1

ML_DIR="$WORKSPACE/quantx/ml"
[ -d "$ML_DIR" ] || fail "Diretório $ML_DIR não encontrado"

cd "$ML_DIR"
pip install -r requirements.txt -q 2>/dev/null

MODEL_DIR="${MODEL_DIR:-/workspace/models}"
mkdir -p "$MODEL_DIR"

MODEL_DIR="$MODEL_DIR" \
SUPABASE_URL="$SUPABASE_URL" \
SUPABASE_SERVICE_ROLE_KEY="$SUPABASE_SERVICE_ROLE_KEY" \
nohup python -m uvicorn main:app \
  --host 0.0.0.0 --port "$ML_PORT" \
  --log-level info \
  > "$LOG_DIR/ml.log" 2>&1 &
ML_PID=$!
echo "$ML_PID" > /tmp/quantx-ml.pid

sleep 5
if curl -sf "http://localhost:$ML_PORT/health" > /dev/null; then
  ok "ML service rodando (PID $ML_PID, porta $ML_PORT)"
else
  fail "ML service não respondeu — veja $LOG_DIR/ml.log"
fi

# ── 6. Backend (Node.js/Fastify) ──────────────────────────────────────────────
info "Iniciando backend..."

pkill -f "tsx src/api/server" 2>/dev/null || true
sleep 1

BACKEND_DIR="$WORKSPACE/quantx/backend"
[ -d "$BACKEND_DIR" ] || fail "Diretório $BACKEND_DIR não encontrado"

cd "$BACKEND_DIR"
npm install -q 2>/dev/null

REDIS_URL="redis://localhost:$REDIS_PORT" \
ML_BASE_URL="http://localhost:$ML_PORT" \
DATABASE_URL="$DATABASE_URL" \
SUPABASE_URL="$SUPABASE_URL" \
SUPABASE_SERVICE_ROLE_KEY="$SUPABASE_SERVICE_ROLE_KEY" \
MODEL_VERSION="$MODEL_VERSION" \
RENDER_DPI="$RENDER_DPI" \
PORT="$BACKEND_PORT" \
nohup npx tsx src/api/server.ts \
  > "$LOG_DIR/backend.log" 2>&1 &
BACKEND_PID=$!
echo "$BACKEND_PID" > /tmp/quantx-backend.pid

sleep 5
if curl -sf "http://localhost:$BACKEND_PORT/health" > /dev/null; then
  ok "Backend rodando (PID $BACKEND_PID, porta $BACKEND_PORT)"
else
  fail "Backend não respondeu — veja $LOG_DIR/backend.log"
fi

# ── 7. Smoke test end-to-end ──────────────────────────────────────────────────
info "Smoke test: worker conecta ao banco..."
if curl -sf "http://localhost:$BACKEND_PORT/health" | python3 -c "import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get('database')=='ok' else 1)" 2>/dev/null; then
  ok "Worker ↔ Banco OK"
else
  info "Health check: banco ainda inicializando — verifique $LOG_DIR/backend.log"
fi

# ── 8. Resumo ─────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                  QuantX está rodando!                   ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  Backend API  : http://localhost:%-26s║\n" "$BACKEND_PORT"
printf "║  ML Service   : http://localhost:%-26s║\n" "$ML_PORT"
printf "║  Redis        : localhost:%-34s║\n" "$REDIS_PORT"
printf "║  Supabase DB  : %-43s║\n" "$(echo "$DATABASE_URL" | sed 's/postgresql:\/\/[^@]*@/postgresql:\/\/***@/')"
printf "║  Storage      : %-43s║\n" "${SUPABASE_URL:-n/a}"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Logs: /tmp/quantx-logs/                                ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Teste:                                                  ║"
echo "║    curl http://localhost:$BACKEND_PORT/health                   ║"
echo "║    curl -X POST http://localhost:$BACKEND_PORT/plans \\          ║"
echo "║      -F file=@sua_planta.pdf                             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
