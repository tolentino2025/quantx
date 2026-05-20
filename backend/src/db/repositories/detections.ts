import { pool } from '../pool.js';
import type { RoutedDetection } from '../../types/index.js';

export async function saveDetections(
  planId: string,
  tenantId: string,
  pageNumber: number,
  detections: RoutedDetection[],
): Promise<void> {
  if (detections.length === 0) return;

  const values: unknown[] = [];
  const placeholders = detections.map((d, i) => {
    const base = i * 14;
    values.push(
      d.detection_id,
      planId,
      tenantId,
      pageNumber,
      d.tile_id,
      d.class_slug,
      d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2,
      d.confidence_final,
      d.sources.yolo?.confidence ?? null,
      d.sources.dinov2?.confidence ?? null,
      d.source_agreement,
      d.router_decision,
      d.router_justification,
      d.border_detection,
      d.cost_cents,
    );
    // 18 params per row
    const p = (n: number) => `$${base + n}`;
    return `(${[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18].map(p).join(',')})`;
  });

  // recalculate with correct base
  const vals2: unknown[] = [];
  const ph2 = detections.map((d, i) => {
    const base = i * 18;
    vals2.push(
      d.detection_id, planId, tenantId, pageNumber, d.tile_id,
      d.class_slug,
      d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2,
      d.confidence_final,
      d.sources.yolo?.confidence ?? null,
      d.sources.dinov2?.confidence ?? null,
      d.source_agreement,
      d.router_decision,
      d.router_justification,
      d.border_detection,
      d.cost_cents,
    );
    return `(${Array.from({ length: 18 }, (_, j) => `$${base + j + 1}`).join(',')})`;
  });

  await pool.query(
    `INSERT INTO detections
       (detection_id, plan_id, tenant_id, page_number, tile_id,
        class_slug, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
        confidence_final, source_yolo, source_dinov2, source_agreement,
        router_decision, router_justification, border_detection, cost_cents)
     VALUES ${ph2.join(',')}
     ON CONFLICT (detection_id) DO NOTHING`,
    vals2,
  );
}

export async function getDetectionsByPlan(
  planId: string,
  decision?: string,
): Promise<Record<string, unknown>[]> {
  const params: unknown[] = [planId];
  const filter = decision ? ' AND router_decision = $2' : '';
  if (decision) params.push(decision);

  const { rows } = await pool.query(
    `SELECT * FROM detections WHERE plan_id = $1${filter} ORDER BY page_number, class_slug`,
    params,
  );
  return rows;
}
