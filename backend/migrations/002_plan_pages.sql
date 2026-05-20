-- 002_plan_pages.sql
-- Tracks individual rendered pages per plan, with Supabase Storage path

CREATE TABLE IF NOT EXISTS plan_pages (
  page_id      UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
  plan_id      UUID        NOT NULL REFERENCES plans(plan_id) ON DELETE CASCADE,
  tenant_id    TEXT        NOT NULL,
  page_number  INTEGER     NOT NULL,
  storage_path TEXT        NOT NULL,
  status       TEXT        NOT NULL DEFAULT 'pending'
               CHECK (status IN ('pending','processing','completed','failed')),
  job_id       TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (plan_id, page_number)
);

CREATE INDEX IF NOT EXISTS plan_pages_plan_id_idx   ON plan_pages (plan_id);
CREATE INDEX IF NOT EXISTS plan_pages_tenant_id_idx ON plan_pages (tenant_id);
