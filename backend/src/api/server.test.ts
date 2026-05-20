import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// ── mocks (must be declared before imports that use them) ─────────────────────

vi.mock('ioredis', () => {
  const Redis = vi.fn().mockImplementation(() => ({
    disconnect: vi.fn(),
  }));
  return { Redis, default: Redis };
});

vi.mock('bullmq', () => {
  const mockJob = {
    id: 'job-123',
    data: { plan_id: 'plan-1', page_number: 1 },
    progress: 0,
    timestamp: Date.now(),
    attemptsMade: 0,
    failedReason: undefined,
    returnvalue: null,
    getState: vi.fn().mockResolvedValue('waiting'),
  };

  const Queue = vi.fn().mockImplementation(() => ({
    add: vi.fn().mockResolvedValue(mockJob),
    getJob: vi.fn().mockResolvedValue(mockJob),
    close: vi.fn().mockResolvedValue(undefined),
  }));
  const Worker = vi.fn().mockImplementation(() => ({ on: vi.fn(), close: vi.fn() }));
  return { Queue, Worker, QueueEvents: vi.fn() };
});

vi.mock('../workers/ml-client.js', () => {
  const MLClient = vi.fn().mockImplementation(() => ({
    health: vi.fn().mockResolvedValue(true),
  }));
  return { MLClient };
});

vi.mock('../recognition/pipeline.js', () => {
  const RecognitionPipeline = vi.fn().mockImplementation(() => ({
    run: vi.fn().mockResolvedValue({
      plan_id: 'plan-1',
      page_number: 1,
      model_version: 'v1',
      scale: '1:100',
      scale_confidence: 0.9,
      detections: [],
      stats: { raw_yolo: 0, after_nms: 0, auto_accepted: 0, verify_with_vision: 0, send_to_review: 0, rejected: 0, vision_cost_cents: 0 },
      warnings: [],
      duration_ms: 100,
    }),
  }));
  return { RecognitionPipeline };
});

vi.mock('../db/pool.js', () => ({
  dbHealthCheck: vi.fn().mockResolvedValue(true),
  pool: {},
}));

vi.mock('../db/repositories/plans.js', () => ({
  createPlan: vi.fn().mockResolvedValue(null),
  updatePlanStatus: vi.fn().mockResolvedValue(undefined),
  listPlansByTenant: vi.fn().mockResolvedValue([]),
}));

vi.mock('../db/repositories/plan-pages.js', () => ({
  createPlanPages: vi.fn().mockResolvedValue([]),
}));

vi.mock('../db/repositories/detections.js', () => ({
  saveDetections: vi.fn().mockResolvedValue(undefined),
}));

vi.mock('../storage/supabase-storage.js', () => ({
  BUCKETS: {
    PLANS_ORIGINAL: 'plans-original',
    PLANS_PAGES: 'plans-pages',
    PLANS_PROCESSED: 'plans-processed',
  },
  uploadBuffer: vi.fn().mockResolvedValue(undefined),
  downloadBuffer: vi.fn().mockResolvedValue(Buffer.from('')),
  getSignedUrl: vi.fn().mockResolvedValue('https://signed.example/pdf'),
  ensureBuckets: vi.fn().mockResolvedValue({ created: [], existing: [] }),
}));

import { buildApp } from './server.js';
import type { FastifyInstance } from 'fastify';
import { Queue } from 'bullmq';

// ── helpers ───────────────────────────────────────────────────────────────────

const VALID_BODY = {
  page_number: 1,
  tenant_id: 'tenant-1',
  storage_path: 'tenant-1/plan-1/page-001.png',
  model_version: 'yolov15-spci-2026.04',
};

// ── tests ─────────────────────────────────────────────────────────────────────

describe('API server', () => {
  let app: FastifyInstance;

  beforeEach(async () => {
    app = await buildApp({ startWorker: false });
    await app.ready();
  });

  afterEach(async () => {
    await app.close();
  });

  // ── health ──────────────────────────────────────────────────────────────────

  it('GET /health returns 200 with status ok', async () => {
    const res = await app.inject({ method: 'GET', url: '/health' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.status).toBe('ok');
    expect(body.service).toBe('quantx-backend');
  });

  // ── POST /plans/:plan_id/recognize ──────────────────────────────────────────

  it('POST /plans/:plan_id/recognize enqueues job and returns 202', async () => {
    const res = await app.inject({
      method: 'POST',
      url: '/plans/plan-abc/recognize',
      payload: VALID_BODY,
    });
    expect(res.statusCode).toBe(202);
    const body = res.json();
    expect(body.status).toBe('queued');
    expect(body.plan_id).toBe('plan-abc');
    expect(body.page_number).toBe(1);
    expect(body.job_id).toBeDefined();
  });

  it('POST calls queue.add once', async () => {
    await app.inject({
      method: 'POST',
      url: '/plans/plan-xyz/recognize',
      payload: VALID_BODY,
    });
    const queueInstance = (Queue as unknown as ReturnType<typeof vi.fn>).mock.results.at(-1)!.value;
    expect(queueInstance.add).toHaveBeenCalledTimes(1);
  });

  it('POST with optional fields accepted', async () => {
    const res = await app.inject({
      method: 'POST',
      url: '/plans/plan-1/recognize',
      payload: { ...VALID_BODY, audit_mode: true, budget_cents: 30, tenant_library_size: 5 },
    });
    expect(res.statusCode).toBe(202);
  });

  it('POST returns 400 when required field missing', async () => {
    const res = await app.inject({
      method: 'POST',
      url: '/plans/plan-1/recognize',
      payload: { page_number: 1, tenant_id: 'tenant-1' }, // missing storage_path and model_version
    });
    expect(res.statusCode).toBe(400);
  });

  it('POST returns 400 when page_number is zero', async () => {
    const res = await app.inject({
      method: 'POST',
      url: '/plans/plan-1/recognize',
      payload: { ...VALID_BODY, page_number: 0 },
    });
    expect(res.statusCode).toBe(400);
  });

  // ── GET /jobs/:job_id ───────────────────────────────────────────────────────

  it('GET /jobs/:job_id returns job state', async () => {
    const res = await app.inject({ method: 'GET', url: '/jobs/job-123' });
    expect(res.statusCode).toBe(200);
    const body = res.json();
    expect(body.job_id).toBe('job-123');
    expect(body.state).toBeDefined();
  });

  it('GET /jobs/:job_id returns 404 for unknown job', async () => {
    const queueInstance = (Queue as unknown as ReturnType<typeof vi.fn>).mock.results.at(-1)!.value;
    queueInstance.getJob.mockResolvedValueOnce(null);

    const res = await app.inject({ method: 'GET', url: '/jobs/nonexistent' });
    expect(res.statusCode).toBe(404);
    expect(res.json().error).toBe('job_not_found');
  });
});
