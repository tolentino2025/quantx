import { pool } from '../pool.js';

export interface PlanRow {
  plan_id: string;
  tenant_id: string;
  file_name: string | null;
  page_count: number | null;
  status: 'processing' | 'completed' | 'partial' | 'failed';
  created_at: Date;
  updated_at: Date;
}

export async function createPlan(params: {
  plan_id: string;
  tenant_id: string;
  file_name?: string;
  page_count?: number;
}): Promise<PlanRow> {
  const { rows } = await pool.query<PlanRow>(
    `INSERT INTO plans (plan_id, tenant_id, file_name, page_count, status)
     VALUES ($1, $2, $3, $4, 'processing')
     RETURNING *`,
    [params.plan_id, params.tenant_id, params.file_name ?? null, params.page_count ?? null],
  );
  return rows[0];
}

export async function getPlan(planId: string): Promise<PlanRow | null> {
  const { rows } = await pool.query<PlanRow>(
    'SELECT * FROM plans WHERE plan_id = $1',
    [planId],
  );
  return rows[0] ?? null;
}

export async function updatePlanStatus(
  planId: string,
  status: PlanRow['status'],
): Promise<void> {
  await pool.query(
    `UPDATE plans SET status = $1, updated_at = NOW() WHERE plan_id = $2`,
    [status, planId],
  );
}

export async function listPlansByTenant(
  tenantId: string,
  limit = 20,
  offset = 0,
): Promise<PlanRow[]> {
  const { rows } = await pool.query<PlanRow>(
    `SELECT * FROM plans WHERE tenant_id = $1
     ORDER BY created_at DESC LIMIT $2 OFFSET $3`,
    [tenantId, limit, offset],
  );
  return rows;
}
