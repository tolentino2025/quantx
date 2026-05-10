import { Worker, Job, Queue, QueueEvents } from 'bullmq';
import type { Redis } from 'ioredis';
import { RecognitionPipeline, type PipelineInput, type PipelineResult } from '../recognition/pipeline.js';
import { MLClient, type MLClientOptions } from './ml-client.js';

export interface RecognitionJobData {
  plan_id: string;
  page_number: number;
  tenant_id: string;
  image_path: string;
  model_version: string;
  tenant_library_size: number;
  audit_mode?: boolean;
  budget_cents?: number;
}

export interface RecognitionJobResult {
  plan_id: string;
  page_number: number;
  model_version: string;
  scale: string;
  scale_confidence: number;
  detection_count: number;
  stats: PipelineResult['stats'];
  warnings: string[];
  duration_ms: number;
}

export const RECOGNITION_QUEUE_NAME = 'recognition:plan';

export function createRecognitionQueue(connection: Redis): Queue<RecognitionJobData, RecognitionJobResult> {
  return new Queue(RECOGNITION_QUEUE_NAME, {
    connection,
    defaultJobOptions: {
      attempts: 3,
      backoff: { type: 'exponential', delay: 2_000 },
      removeOnComplete: { age: 86_400, count: 1_000 },
      removeOnFail: { age: 604_800, count: 500 },
    },
  });
}

export function createRecognitionWorker(
  connection: Redis,
  mlClientOptions: MLClientOptions,
  concurrency = 2,
): Worker<RecognitionJobData, RecognitionJobResult> {
  const mlClient = new MLClient(mlClientOptions);
  const pipeline = new RecognitionPipeline(mlClient);

  const worker = new Worker<RecognitionJobData, RecognitionJobResult>(
    RECOGNITION_QUEUE_NAME,
    async (job: Job<RecognitionJobData, RecognitionJobResult>) => {
      const { data } = job;

      await job.updateProgress(5);
      job.log(`Starting recognition | plan=${data.plan_id} page=${data.page_number} tenant=${data.tenant_id}`);

      const input: PipelineInput = {
        plan_id: data.plan_id,
        page_number: data.page_number,
        tenant_id: data.tenant_id,
        image_path: data.image_path,
        model_version: data.model_version,
        tenant_library_size: data.tenant_library_size,
        audit_mode: data.audit_mode,
        budget_cents: data.budget_cents,
      };

      await job.updateProgress(10);
      const result = await pipeline.run(input);

      await job.updateProgress(95);
      job.log(
        `Recognition done | plan=${result.plan_id} detections=${result.detections.length} ` +
        `scale=${result.scale} duration=${result.duration_ms}ms warnings=${result.warnings.length}`,
      );

      if (result.warnings.length > 0) {
        job.log(`Warnings: ${result.warnings.join(' | ')}`);
      }

      await job.updateProgress(100);

      return {
        plan_id: result.plan_id,
        page_number: result.page_number,
        model_version: result.model_version,
        scale: result.scale,
        scale_confidence: result.scale_confidence,
        detection_count: result.detections.length,
        stats: result.stats,
        warnings: result.warnings,
        duration_ms: result.duration_ms,
      };
    },
    {
      connection,
      concurrency,
      limiter: { max: 10, duration: 60_000 },
    },
  );

  worker.on('completed', (job, result) => {
    console.info(
      `[recognition-worker] completed | job=${job.id} plan=${result.plan_id} ` +
      `detections=${result.detection_count} duration=${result.duration_ms}ms`,
    );
  });

  worker.on('failed', (job, err) => {
    console.error(
      `[recognition-worker] failed | job=${job?.id} plan=${job?.data.plan_id} ` +
      `attempt=${job?.attemptsMade} error=${err.message}`,
    );
  });

  worker.on('stalled', (jobId) => {
    console.warn(`[recognition-worker] stalled | job=${jobId}`);
  });

  worker.on('error', (err) => {
    console.error(`[recognition-worker] worker error | ${err.message}`);
  });

  return worker;
}
