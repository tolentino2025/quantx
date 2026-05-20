import { pool } from '../pool.js';

export interface PlanPageRow {
  page_id: string;
  plan_id: string;
  tenant_id: string;
  page_number: number;
  storage_path: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  job_id: string | null;
  created_at: Date;
  updated_at: Date;
}

export async function createPlanPages(
  pages: Array<{
    plan_id: string;
    tenant_id: string;
    page_number: number;
    storage_path: string;
  }>,
): Promise<PlanPageRow[]> {
  if (pages.length === 0) return [];

  const vals: unknown[] = [];
  const ph = pages.map((p, i) => {
    vals.push(p.plan_id, p.tenant_id, p.page_number, p.storage_path);
    const b = i * 4;
    return `($${b + 1},$${b + 2},$${b + 3},$${b + 4})`;
  });

  const { rows } = await pool.query<PlanPageRow>(
    `INSERT INTO plan_pages (plan_id, tenant_id, page_number, storage_path)
     VALUES ${ph.join(',')} RETURNING *`,
    vals,
  );
  return rows;
}

export async function updatePageStatus(
  planId: string,
  pageNumber: number,
  status: PlanPageRow['status'],
  jobId?: string,
): Promise<void> {
  await pool.query(
    `UPDATE plan_pages
     SET status = $1, job_id = COALESCE($2, job_id), updated_at = NOW()
     WHERE plan_id = $3 AND page_number = $4`,
    [status, jobId ?? null, planId, pageNumber],
  );
}

export async function getPlanPages(planId: string): Promise<PlanPageRow[]> {
  const { rows } = await pool.query<PlanPageRow>(
    'SELECT * FROM plan_pages WHERE plan_id = $1 ORDER BY page_number',
    [planId],
  );
  return rows;
}
