import Fastify, { type FastifyInstance } from 'fastify';
import { Redis } from 'ioredis';
import { randomUUID } from 'crypto';
import {
  createRecognitionQueue,
  createRecognitionWorker,
  RECOGNITION_QUEUE_NAME,
  type RecognitionJobData,
} from '../workers/recognition-worker.js';
import { MLClient } from '../workers/ml-client.js';

// ── env ───────────────────────────────────────────────────────────────────────

const ML_BASE_URL = process.env.ML_BASE_URL ?? 'http://localhost:8000';
const REDIS_URL   = process.env.REDIS_URL   ?? 'redis://localhost:6379';
const PORT        = Number(process.env.PORT ?? 3000);
const HOST        = process.env.HOST        ?? '0.0.0.0';

// ── app factory ───────────────────────────────────────────────────────────────

export async function buildApp(opts: {
  redisUrl?: string;
  mlBaseUrl?: string;
  startWorker?: boolean;
} = {}): Promise<FastifyInstance> {
  const redisUrl  = opts.redisUrl  ?? REDIS_URL;
  const mlBaseUrl = opts.mlBaseUrl ?? ML_BASE_URL;

  const app = Fastify({ logger: { level: 'info' } });

  const redis = new Redis(redisUrl, { maxRetriesPerRequest: null });
  const queue = createRecognitionQueue(redis);

  if (opts.startWorker ?? true) {
    createRecognitionWorker(redis, { baseUrl: mlBaseUrl });
  }

  // ── health ──────────────────────────────────────────────────────────────────

  app.get('/health', async () => {
    const mlClient = new MLClient({ baseUrl: mlBaseUrl, timeoutMs: 3_000, retries: 0 });
    const mlOk = await mlClient.health();
    return {
      status: mlOk ? 'ok' : 'degraded',
      service: 'quantx-backend',
      ml_service: mlOk ? 'ok' : 'unreachable',
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
