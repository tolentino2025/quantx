#!/usr/bin/env bash
# setup-runpod.sh — configura e sobe todos os serviços QuantX num RunPod sem Docker
# Uso: bash setup-runpod.sh
# Requer: RunPod com Ubuntu, acesso root

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

# Defaults
PG_USER="${POSTGRES_USER:-takeoff}"
PG_PASS="${POSTGRES_PASSWORD:-quantx_$(openssl rand -hex 8)}"
PG_DB="${POSTGRES_DB:-takeoffdb}"
REDIS_PORT="${REDIS_PORT:-6379}"
ML_PORT="${ML_PORT:-8000}"
BACKEND_PORT="${BACKEND_PORT:-3000}"
MODEL_VERSION="${MODEL_VERSION:-yolov15-spci-2026.04}"
RENDER_DPI="${RENDER_DPI:-150}"

# Carregar .env se existir
if [ -f "$ENV_FILE" ]; then
  set -a; source "$ENV_FILE"; set +a
  ok ".env carregado de $ENV_FILE"
fi

# Reaplica defaults (podem ter sido sobrescritos por .env)
PG_USER="${POSTGRES_USER:-$PG_USER}"
PG_PASS="${POSTGRES_PASSWORD:-$PG_PASS}"
PG_DB="${POSTGRES_DB:-$PG_DB}"
DATABASE_URL="${DATABASE_URL:-postgresql://$PG_USER:$PG_PASS@localhost:5432/$PG_DB}"

# Grava DATABASE_URL no .env se não existir
if ! grep -q "DATABASE_URL" "$ENV_FILE" 2>/dev/null; then
  echo "DATABASE_URL=$DATABASE_URL" >> "$ENV_FILE"
  ok "DATABASE_URL gravado em $ENV_FILE"
fi

# ── 2. Dependências de sistema ────────────────────────────────────────────────
info "Verificando dependências do sistema..."

apt-get update -qq 2>/dev/null

PKGS=()
command -v redis-server &>/dev/null || PKGS+=(redis-server)
command -v psql         &>/dev/null || PKGS+=(postgresql postgresql-contrib)
command -v pdftoppm     &>/dev/null || PKGS+=(poppler-utils)
command -v tesseract    &>/dev/null || PKGS+=(tesseract-ocr tesseract-ocr-por)
command -v curl         &>/dev/null || PKGS+=(curl)

if [ ${#PKGS[@]} -gt 0 ]; then
  info "Instalando: ${PKGS[*]}"
  apt-get install -y --no-install-recommends "${PKGS[@]}" -qq 2>/dev/null
fi

# pgvector
if ! psql -U "$PG_USER" -c "SELECT extname FROM pg_extension WHERE extname='vector'" 2>/dev/null | grep -q vector; then
  if ! apt-cache show postgresql-16-pgvector &>/dev/null; then
    apt-get install -y --no-install-recommends postgresql-common -qq
    /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y -q 2>/dev/null || true
  fi
  apt-get install -y --no-install-recommends "postgresql-$(pg_lsclusters -h | awk '{print $1}' | tail -1)-pgvector" -qq 2>/dev/null \
    || info "pgvector não instalado via apt — tentando build do fonte"
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

# ── 4. PostgreSQL ─────────────────────────────────────────────────────────────

# Se DATABASE_URL aponta para host externo (Supabase, RDS, etc.) pular instalação local
_db_host=$(echo "$DATABASE_URL" | sed -E 's|.*@([^:/]+).*|\1|')
_is_external_db=false
if [[ "$DATABASE_URL" != *"localhost"* && "$DATABASE_URL" != *"127.0.0.1"* && -n "$DATABASE_URL" ]]; then
  _is_external_db=true
fi

if $_is_external_db; then
  info "DATABASE_URL aponta para host externo ($_db_host) — pulando instalação local do PostgreSQL"
  # Apenas aplica migrations no banco remoto
  MIGRATIONS_DIR="$WORKSPACE/quantx/backend/migrations"
  if [ -d "$MIGRATIONS_DIR" ]; then
    info "Aplicando migrations no banco remoto..."
    for f in "$MIGRATIONS_DIR"/*.sql; do
      info "  $(basename $f)"
      psql "$DATABASE_URL" -f "$f" >> "$LOG_DIR/migrations.log" 2>&1 \
        && ok "  $(basename $f)" || info "  $(basename $f) — verifique $LOG_DIR/migrations.log"
    done
  fi
  ok "Banco externo configurado ($DATABASE_URL)"
else
  info "Configurando PostgreSQL local..."

  PG_VERSION=$(pg_lsclusters -h 2>/dev/null | awk '{print $1}' | sort -n | tail -1)
  PG_CLUSTER=$(pg_lsclusters -h 2>/dev/null | awk '{print $2}' | tail -1)

  if [ -z "$PG_VERSION" ]; then
    fail "PostgreSQL não encontrado — instale manualmente: apt-get install postgresql"
  fi

  # Inicia o cluster se não estiver rodando
  if ! pg_lsclusters -h | grep -q "online"; then
    pg_ctlcluster "$PG_VERSION" "$PG_CLUSTER" start
    sleep 2
  fi

  # Cria usuário e banco
  su -c "psql -tc \"SELECT 1 FROM pg_roles WHERE rolname='$PG_USER'\" | grep -q 1 || \
    psql -c \"CREATE USER $PG_USER WITH PASSWORD '$PG_PASS' CREATEDB;\"" postgres 2>/dev/null || true

  su -c "psql -tc \"SELECT 1 FROM pg_database WHERE datname='$PG_DB'\" | grep -q 1 || \
    createdb -O $PG_USER $PG_DB" postgres 2>/dev/null || true

  # Extensões
  su -c "psql -d $PG_DB -c 'CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";'" postgres 2>/dev/null || true
  su -c "psql -d $PG_DB -c 'CREATE EXTENSION IF NOT EXISTS vector;'" postgres 2>/dev/null \
    && ok "pgvector habilitado" || info "pgvector não disponível — embeddings usarão fallback em memória"

  # Migrations
  MIGRATIONS_DIR="$WORKSPACE/quantx/backend/migrations"
  if [ -d "$MIGRATIONS_DIR" ]; then
    for f in "$MIGRATIONS_DIR"/*.sql; do
      info "Aplicando migration: $(basename $f)"
      PGPASSWORD="$PG_PASS" psql -h localhost -U "$PG_USER" -d "$PG_DB" -f "$f" \
        >> "$LOG_DIR/migrations.log" 2>&1 && ok "$(basename $f)" || info "$(basename $f) — verifique $LOG_DIR/migrations.log"
    done
  else
    info "Pasta de migrations não encontrada em $MIGRATIONS_DIR"
  fi

  ok "PostgreSQL local pronto ($PG_USER@localhost:5432/$PG_DB)"
fi

# ── 5. ML Service (Python/FastAPI) ────────────────────────────────────────────
info "Iniciando ML service..."

# Para instância anterior se existir
pkill -f "uvicorn main:app" 2>/dev/null || true
sleep 1

ML_DIR="$WORKSPACE/quantx/ml"
[ -d "$ML_DIR" ] || fail "Diretório $ML_DIR não encontrado"

cd "$ML_DIR"
pip install -r requirements.txt -q 2>/dev/null

MODEL_DIR="${MODEL_DIR:-/workspace/models}"
mkdir -p "$MODEL_DIR"

MODEL_DIR="$MODEL_DIR" nohup python -m uvicorn main:app \
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
UPLOAD_DIR="/tmp/quantx-pages" \
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

# ── 7. Resumo ─────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                  QuantX está rodando!                   ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  Backend API  : http://localhost:%-26s║\n" "$BACKEND_PORT"
printf "║  ML Service   : http://localhost:%-26s║\n" "$ML_PORT"
printf "║  Redis        : localhost:%-34s║\n" "$REDIS_PORT"
printf "║  PostgreSQL   : localhost:5432/%-29s║\n" "$PG_DB"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Logs: /tmp/quantx-logs/                                ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Teste:                                                  ║"
echo "║    curl http://localhost:$BACKEND_PORT/health                   ║"
echo "║    curl -X POST http://localhost:$BACKEND_PORT/plans \\          ║"
echo "║      -F file=@sua_planta.pdf                             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
