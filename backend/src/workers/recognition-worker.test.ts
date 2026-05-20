import { describe, it, expect, vi, beforeEach } from 'vitest';

// BullMQ is mocked so tests run without Redis
vi.mock('bullmq', () => {
  const Worker = vi.fn().mockImplementation((_name, processor, _opts) => ({
    processor,
    on: vi.fn(),
    close: vi.fn(),
  }));
  const Queue = vi.fn().mockImplementation(() => ({
    add: vi.fn(),
    on: vi.fn(),
    close: vi.fn(),
  }));
  return { Worker, Queue, QueueEvents: vi.fn() };
});

vi.mock('../recognition/pipeline.js', () => {
  const run = vi.fn().mockResolvedValue({
    plan_id: 'plan-1',
    page_number: 1,
    model_version: 'yolov15-spci-2026.04',
    scale: '1:100',
    scale_confidence: 0.92,
    detections: [{ detection_id: 'd1', router_decision: 'auto_accept' }],
    stats: {
      raw_yolo: 1,
      after_nms: 1,
      auto_accepted: 1,
      verify_with_vision: 0,
      send_to_review: 0,
      rejected: 0,
      vision_cost_cents: 0,
    },
    warnings: [],
    duration_ms: 312,
  });
  const RecognitionPipeline = vi.fn().mockImplementation(() => ({ run }));
  return { RecognitionPipeline };
});

vi.mock('fs/promises', () => ({
  writeFile: vi.fn().mockResolvedValue(undefined),
  rm: vi.fn().mockResolvedValue(undefined),
  mkdir: vi.fn().mockResolvedValue(undefined),
}));

vi.mock('../db/repositories/detections.js', () => ({
  saveDetections: vi.fn().mockResolvedValue(undefined),
}));

vi.mock('../db/repositories/plans.js', () => ({
  updatePlanStatus: vi.fn().mockResolvedValue(undefined),
}));

vi.mock('../db/repositories/plan-pages.js', () => ({
  updatePageStatus: vi.fn().mockResolvedValue(undefined),
}));

vi.mock('../storage/supabase-storage.js', () => ({
  BUCKETS: {
    PLANS_ORIGINAL: 'plans-original',
    PLANS_PAGES: 'plans-pages',
    PLANS_PROCESSED: 'plans-processed',
  },
  downloadBuffer: vi.fn().mockResolvedValue(Buffer.from('fake-png')),
  uploadBuffer: vi.fn().mockResolvedValue(undefined),
}));

import { createRecognitionWorker, createRecognitionQueue, RECOGNITION_QUEUE_NAME } from './recognition-worker.js';
import { Worker, Queue } from 'bullmq';
import { RecognitionPipeline } from '../recognition/pipeline.js';

const fakeRedis = {} as any;
const mlOpts = { baseUrl: 'http://localhost:8000' };

describe('createRecognitionQueue', () => {
  it('creates a Queue with the correct name', () => {
    createRecognitionQueue(fakeRedis);
    expect(Queue).toHaveBeenCalledWith(RECOGNITION_QUEUE_NAME, expect.any(Object));
  });
});

describe('createRecognitionWorker', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('creates a Worker bound to RECOGNITION_QUEUE_NAME', () => {
    createRecognitionWorker(fakeRedis, mlOpts);
    expect(Worker).toHaveBeenCalledWith(RECOGNITION_QUEUE_NAME, expect.any(Function), expect.any(Object));
  });

  it('instantiates RecognitionPipeline once', () => {
    createRecognitionWorker(fakeRedis, mlOpts);
    expect(RecognitionPipeline).toHaveBeenCalledTimes(1);
  });

  it('registers event listeners on the worker', () => {
    const worker = createRecognitionWorker(fakeRedis, mlOpts);
    expect(worker.on).toHaveBeenCalledWith('completed', expect.any(Function));
    expect(worker.on).toHaveBeenCalledWith('failed', expect.any(Function));
    expect(worker.on).toHaveBeenCalledWith('stalled', expect.any(Function));
    expect(worker.on).toHaveBeenCalledWith('error', expect.any(Function));
  });

  it('processor calls pipeline.run with correct PipelineInput', async () => {
    createRecognitionWorker(fakeRedis, mlOpts);

    const [, processor] = (Worker as unknown as ReturnType<typeof vi.fn>).mock.calls[0];

    const job = {
      id: 'job-1',
      data: {
        plan_id: 'plan-1',
        page_number: 1,
        tenant_id: 'tenant-1',
        storage_path: 'tenant-1/plan-1/page-001.png',
        model_version: 'yolov15-spci-2026.04',
        tenant_library_size: 0,
      },
      updateProgress: vi.fn().mockResolvedValue(undefined),
      log: vi.fn(),
      attemptsMade: 0,
    };

    const result = await processor(job);

    const pipelineInstance = (RecognitionPipeline as unknown as ReturnType<typeof vi.fn>).mock.results[0].value;
    expect(pipelineInstance.run).toHaveBeenCalledWith(
      expect.objectContaining({
        plan_id: 'plan-1',
        page_number: 1,
        tenant_id: 'tenant-1',
        image_path: expect.stringContaining('page-001.png'),
      }),
    );
    expect(result.plan_id).toBe('plan-1');
    expect(result.detection_count).toBe(1);
  });

  it('processor result contains stats and warnings', async () => {
    createRecognitionWorker(fakeRedis, mlOpts);
    const [, processor] = (Worker as unknown as ReturnType<typeof vi.fn>).mock.calls[0];

    const job = {
      id: 'job-2',
      data: {
        plan_id: 'plan-1',
        page_number: 1,
        tenant_id: 'tenant-1',
        storage_path: 'tenant-1/plan-1/page-001.png',
        model_version: 'yolov15-spci-2026.04',
        tenant_library_size: 0,
      },
      updateProgress: vi.fn().mockResolvedValue(undefined),
      log: vi.fn(),
    };

    const result = await processor(job);
    expect(result.stats).toBeDefined();
    expect(result.warnings).toBeInstanceOf(Array);
    expect(result.duration_ms).toBe(312);
  });

  it('respects custom concurrency option', () => {
    createRecognitionWorker(fakeRedis, mlOpts, 4);
    const [, , opts] = (Worker as unknown as ReturnType<typeof vi.fn>).mock.calls[0];
    expect(opts.concurrency).toBe(4);
  });
});
