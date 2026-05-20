-- QuantX — schema inicial
-- Roda automaticamente no primeiro `docker compose up` (via entrypoint do Postgres)

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- ── planos ────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS plans (
    plan_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id   TEXT NOT NULL,
    file_name   TEXT,
    page_count  INT,
    status      TEXT NOT NULL DEFAULT 'processing',  -- processing | completed | partial | failed
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_plans_tenant ON plans(tenant_id);

-- ── detecções ─────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS detections (
    detection_id        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plan_id             UUID NOT NULL REFERENCES plans(plan_id) ON DELETE CASCADE,
    tenant_id           TEXT NOT NULL,
    page_number         INT NOT NULL,
    tile_id             TEXT,
    class_slug          TEXT NOT NULL,
    bbox_x1             FLOAT NOT NULL,
    bbox_y1             FLOAT NOT NULL,
    bbox_x2             FLOAT NOT NULL,
    bbox_y2             FLOAT NOT NULL,
    confidence_final    FLOAT,
    source_yolo         FLOAT,
    source_dinov2       FLOAT,
    source_agreement    TEXT,
    router_decision     TEXT NOT NULL,  -- auto_accept | verify_with_vision | send_to_review | reject
    router_justification TEXT,
    model_version       TEXT,
    border_detection    BOOLEAN DEFAULT FALSE,
    cost_cents          FLOAT DEFAULT 0,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_detections_plan    ON detections(plan_id);
CREATE INDEX IF NOT EXISTS idx_detections_tenant  ON detections(tenant_id);
CREATE INDEX IF NOT EXISTS idx_detections_class   ON detections(class_slug);
CREATE INDEX IF NOT EXISTS idx_detections_decision ON detections(router_decision);

-- ── correções humanas ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS corrections (
    correction_id       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detection_id        UUID REFERENCES detections(detection_id) ON DELETE SET NULL,
    plan_id             UUID REFERENCES plans(plan_id) ON DELETE CASCADE,
    tenant_id           TEXT NOT NULL,
    reviewer_id         TEXT NOT NULL,  -- hash anonimizado (LGPD)
    action              TEXT NOT NULL,  -- accept | reject | reclassify | adjust_bbox | add_new
    original_class      TEXT,
    corrected_class     TEXT,
    corrected_bbox_x1   FLOAT,
    corrected_bbox_y1   FLOAT,
    corrected_bbox_x2   FLOAT,
    corrected_bbox_y2   FLOAT,
    reason              TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_corrections_plan   ON corrections(plan_id);
CREATE INDEX IF NOT EXISTS idx_corrections_tenant ON corrections(tenant_id);

-- ── embeddings DINOv2 (pgvector) ──────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS embeddings (
    sample_id   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id   TEXT NOT NULL,
    class_slug  TEXT NOT NULL,
    embedding   vector(768) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    deleted_at  TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_embeddings_tenant ON embeddings(tenant_id) WHERE deleted_at IS NULL;

-- HNSW index para busca aproximada por similaridade cosine
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw
    ON embeddings USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE deleted_at IS NULL;
