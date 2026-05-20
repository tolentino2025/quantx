import Fastify, { type FastifyInstance } from 'fastify';
import multipart from '@fastify/multipart';
import { Redis } from 'ioredis';
import { randomUUID } from 'crypto';
import { createWriteStream, mkdirSync } from 'fs';
import { join } from 'path';
import { pipeline } from 'stream/promises';
import {
  createRecognitionQueue,
  createRecognitionWorker,
  RECOGNITION_QUEUE_NAME,
  type RecognitionJobData,
} from '../workers/recognition-worker.js';
import { MLClient } from '../workers/ml-client.js';
import { dbHealthCheck } from '../db/pool.js';
import { createPlan, getPlan, updatePlanStatus, listPlansByTenant } from '../db/repositories/plans.js';

// ── env ───────────────────────────────────────────────────────────────────────

const ML_BASE_URL  = process.env.ML_BASE_URL  ?? 'http://localhost:8000';
const REDIS_URL    = process.env.REDIS_URL    ?? 'redis://localhost:6379';
const PORT         = Number(process.env.PORT  ?? 3000);
const HOST         = process.env.HOST         ?? '0.0.0.0';
const UPLOAD_DIR   = process.env.UPLOAD_DIR   ?? '/tmp/quantx-pages';
const MODEL_VERSION = process.env.MODEL_VERSION ?? 'yolov15-spci-2026.04';
const RENDER_DPI   = Number(process.env.RENDER_DPI ?? 150);

// ── app factory ───────────────────────────────────────────────────────────────

export async function buildApp(opts: {
  redisUrl?: string;
  mlBaseUrl?: string;
  startWorker?: boolean;
} = {}): Promise<FastifyInstance> {
  const redisUrl  = opts.redisUrl  ?? REDIS_URL;
  const mlBaseUrl = opts.mlBaseUrl ?? ML_BASE_URL;

  const app = Fastify({ logger: { level: 'info' } });

  await app.register(multipart, {
    limits: { fileSize: 200 * 1024 * 1024 }, // 200 MB
  });

  const redis = new Redis(redisUrl, { maxRetriesPerRequest: null });
  const queue = createRecognitionQueue(redis);

  if (opts.startWorker ?? true) {
    createRecognitionWorker(redis, { baseUrl: mlBaseUrl });
  }

  // ── health ──────────────────────────────────────────────────────────────────

  app.get('/health', async () => {
    const mlClient = new MLClient({ baseUrl: mlBaseUrl, timeoutMs: 3_000, retries: 0 });
    const [mlOk, dbOk] = await Promise.all([mlClient.health(), dbHealthCheck()]);
    const status = mlOk && dbOk ? 'ok' : 'degraded';
    return {
      status,
      service: 'quantx-backend',
      ml_service: mlOk ? 'ok' : 'unreachable',
      database: dbOk ? 'ok' : 'unreachable',
    };
  });

  // ── POST /plans — upload PDF, render pages, queue all recognition jobs ───────

  app.post('/plans', async (request, reply) => {
    const data = await request.file();
    if (!data) {
      reply.code(400);
      return { error: 'no_file', detail: 'Envie o PDF como multipart/form-data com o campo "file"' };
    }
    if (!data.mimetype.includes('pdf') && !data.filename?.endsWith('.pdf')) {
      reply.code(422);
      return { error: 'invalid_file_type', detail: 'Apenas arquivos PDF são aceitos' };
    }

    const tenantId     = (request.query as Record<string, string>).tenant_id ?? 'default';
    const planId       = randomUUID();
    const planDir      = join(UPLOAD_DIR, planId);
    const pdfPath      = join(planDir, 'original.pdf');

    mkdirSync(planDir, { recursive: true });
    await pipeline(data.file, createWriteStream(pdfPath));

    // Persiste o plano no banco antes de qualquer processamento
    await createPlan({ plan_id: planId, tenant_id: tenantId, file_name: data.filename }).catch(() => null);

    // Render PDF pages via ML service
    let renderResult: { page_count: number; image_paths: string[] };
    try {
      const res = await fetch(`${mlBaseUrl}/render-pdf`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          plan_id: planId,
          pdf_path: pdfPath,
          dpi: RENDER_DPI,
          output_dir: UPLOAD_DIR,
        }),
        signal: AbortSignal.timeout(120_000),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText })) as { detail?: string };
        reply.code(res.status >= 500 ? 502 : res.status);
        return { error: 'render_failed', detail: err.detail ?? 'ML service error' };
      }
      renderResult = await res.json() as typeof renderResult;
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      reply.code(502);
      return { error: 'render_failed', detail: `ML service unreachable: ${msg}` };
    }

    // Queue one recognition job per page
    const jobs = await Promise.all(
      renderResult.image_paths.map((imagePath, i) => {
        const jobData: RecognitionJobData = {
          plan_id: planId,
          page_number: i + 1,
          tenant_id: tenantId,
          image_path: imagePath,
          model_version: MODEL_VERSION,
          tenant_library_size: 0,
        };
        return queue.add(`recognize-${planId}-p${i + 1}`, jobData, { jobId: randomUUID() });
      }),
    );

    reply.code(202);
    return {
      plan_id: planId,
      tenant_id: tenantId,
      page_count: renderResult.page_count,
      status: 'processing',
      jobs: jobs.map((j, i) => ({ job_id: j.id, page_number: i + 1 })),
    };
  });

  // ── POST /plans/:plan_id/recognize ──────────────────────────────────────────

  interface RecognizeBody {
    page_number: number;
    tenant_id: string;
    image_path: string;
    model_version: string;
    tenant_library_size?: number;
    audit_mode?: boolean;
    budget_cents?: number;
  }

  app.post<{ Params: { plan_id: string }; Body: RecognizeBody }>(
    '/plans/:plan_id/recognize',
    {
      schema: {
        params: {
          type: 'object',
          required: ['plan_id'],
          properties: { plan_id: { type: 'string', minLength: 1 } },
        },
        body: {
          type: 'object',
          required: ['page_number', 'tenant_id', 'image_path', 'model_version'],
          properties: {
            page_number:          { type: 'integer', minimum: 1 },
            tenant_id:            { type: 'string', minLength: 1 },
            image_path:           { type: 'string', minLength: 1 },
            model_version:        { type: 'string', minLength: 1 },
            tenant_library_size:  { type: 'integer', minimum: 0 },
            audit_mode:           { type: 'boolean' },
            budget_cents:         { type: 'number', minimum: 0 },
          },
        },
      },
    },
    async (request, reply) => {
      const { plan_id } = request.params;
      const body = request.body;

      const jobData: RecognitionJobData = {
        plan_id,
        page_number: body.page_number,
        tenant_id: body.tenant_id,
        image_path: body.image_path,
        model_version: body.model_version,
        tenant_library_size: body.tenant_library_size ?? 0,
        audit_mode: body.audit_mode,
        budget_cents: body.budget_cents,
      };

      const job = await queue.add(`recognize-${plan_id}-p${body.page_number}`, jobData, {
        jobId: randomUUID(),
      });

      reply.code(202);
      return {
        job_id: job.id,
        plan_id,
        page_number: body.page_number,
        queue: RECOGNITION_QUEUE_NAME,
        status: 'queued',
      };
    },
  );

  // ── GET /jobs/:job_id ───────────────────────────────────────────────────────

  app.get<{ Params: { job_id: string } }>(
    '/jobs/:job_id',
    {
      schema: {
        params: {
          type: 'object',
          required: ['job_id'],
          properties: { job_id: { type: 'string', minLength: 1 } },
        },
      },
    },
    async (request, reply) => {
      const job = await queue.getJob(request.params.job_id);

      if (!job) {
        reply.code(404);
        return { error: 'job_not_found', job_id: request.params.job_id };
      }

      const state = await job.getState();
      const progress = job.progress;
      const result = state === 'completed' ? await job.returnvalue : undefined;
      const failedReason = state === 'failed' ? job.failedReason : undefined;

      return {
        job_id: job.id,
        plan_id: job.data.plan_id,
        page_number: job.data.page_number,
        state,
        progress,
        ...(result      ? { result }       : {}),
        ...(failedReason ? { error: failedReason } : {}),
        attempts_made: job.attemptsMade,
        created_at: new Date(job.timestamp).toISOString(),
      };
    },
  );

  // ── GET /plans — list plans for a tenant ──────────────────────────────────

  app.get('/plans', async (request, reply) => {
    const { tenant_id, limit, offset } = request.query as Record<string, string>;
    if (!tenant_id) {
      reply.code(400);
      return { error: 'missing_param', detail: 'tenant_id é obrigatório' };
    }
    const rows = await listPlansByTenant(
      tenant_id,
      limit ? Number(limit) : 20,
      offset ? Number(offset) : 0,
    );
    return { tenant_id, plans: rows, count: rows.length };
  });

  // ── GET /plans/:plan_id — poll all page jobs for a plan ────────────────────

  app.get<{ Params: { plan_id: string } }>(
    '/plans/:plan_id',
    async (request, reply) => {
      const { plan_id } = request.params;

      // BullMQ doesn't store job→plan mapping natively; we query by naming convention
      const jobList = await queue.getJobs(['waiting', 'active', 'completed', 'failed', 'delayed']);
      const planJobs = jobList.filter((j) => j.data?.plan_id === plan_id);

      if (planJobs.length === 0) {
        reply.code(404);
        return { error: 'plan_not_found', plan_id };
      }

      const pages = await Promise.all(
        planJobs.map(async (job) => {
          const state = await job.getState();
          return {
            job_id: job.id,
            page_number: job.data.page_number,
            state,
            progress: job.progress,
            ...(state === 'completed' ? { result: await job.returnvalue } : {}),
            ...(state === 'failed'    ? { error: job.failedReason }        : {}),
          };
        }),
      );

      pages.sort((a, b) => a.page_number - b.page_number);

      const allDone = pages.every((p) => p.state === 'completed' || p.state === 'failed');
      const anyFailed = pages.some((p) => p.state === 'failed');
      const derivedStatus = allDone ? (anyFailed ? 'partial' : 'completed') : 'processing';

      if (allDone) {
        await updatePlanStatus(plan_id, derivedStatus as 'completed' | 'partial').catch(() => null);
      }

      return {
        plan_id,
        page_count: pages.length,
        status: derivedStatus,
        pages,
      };
    },
  );

  // ── graceful shutdown ───────────────────────────────────────────────────────

  app.addHook('onClose', async () => {
    await queue.close();
    redis.disconnect();
  });

  return app;
}

// ── entrypoint ────────────────────────────────────────────────────────────────

if (process.argv[1]?.endsWith('server.ts') || process.argv[1]?.endsWith('server.js')) {
  buildApp({ startWorker: true }).then((app) =>
    app.listen({ port: PORT, host: HOST }, (err) => {
      if (err) { app.log.error(err); process.exit(1); }
    }),
  );
}
