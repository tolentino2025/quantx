import { describe, it, expect, vi, beforeEach } from 'vitest';
import { RecognitionPipeline } from './pipeline.js';
import type { MLClient, ScaleDetectionResponse, YOLOInferenceResponse, DINOv2MatchResponse } from '../workers/ml-client.js';

// ── helpers ──────────────────────────────────────────────────────────────────

function makeScaleResult(overrides: Partial<ScaleDetectionResponse> = {}): ScaleDetectionResponse {
  return {
    plan_id: 'plan-1',
    page_number: 1,
    scale: '1:100',
    scale_confidence: 0.92,
    method: 'ocr_cartouche',
    cartouche_region: null,
    legend_region: null,
    recommended: { dpi: 300, tile_size: 1024, overlap: 0.2, imgsz: 1024, dinov2_window: 128 },
    uncertainty_note: '',
    warnings: [],
    ...overrides,
  };
}

function makeYOLOResult(overrides: Partial<YOLOInferenceResponse> = {}): YOLOInferenceResponse {
  return {
    plan_id: 'plan-1',
    page_number: 1,
    model_version: 'yolov15-spci-2026.04',
    total_detections: 2,
    warnings: [],
    tile_results: [
      {
        tile: { tile_id: 'tile-001', origin_x: 0, origin_y: 0, width: 1024, height: 1024 },
        detections: [
          {
            detection_id: 'det-001',
            tile_id: 'tile-001',
            class_slug: 'sprinkler-pendent-k57',
            bbox_tile: { x1: 100, y1: 100, x2: 140, y2: 140 },
            confidence: 0.91,
            source: 'yolo',
            model_version: 'yolov15-spci-2026.04',
          },
          {
            detection_id: 'det-002',
            tile_id: 'tile-001',
            class_slug: 'extintor-pqs',
            bbox_tile: { x1: 500, y1: 500, x2: 540, y2: 540 },
            confidence: 0.75,
            source: 'yolo',
            model_version: 'yolov15-spci-2026.04',
          },
        ],
      },
    ],
    ...overrides,
  };
}

function makeDINOv2Result(overrides: Partial<DINOv2MatchResponse> = {}): DINOv2MatchResponse {
  return {
    tenant_id: 'tenant-1',
    library_size: 10,
    matches: [],
    ...overrides,
  };
}

function makeMockClient(overrides: Partial<MLClient> = {}): MLClient {
  return {
    detectScale: vi.fn().mockResolvedValue(makeScaleResult()),
    runYOLO: vi.fn().mockResolvedValue(makeYOLOResult()),
    matchDINOv2: vi.fn().mockResolvedValue(makeDINOv2Result()),
    health: vi.fn().mockResolvedValue(true),
    ...overrides,
  } as unknown as MLClient;
}

const BASE_INPUT = {
  plan_id: 'plan-1',
  page_number: 1,
  tenant_id: 'tenant-1',
  image_path: '/tmp/test.png',
  model_version: 'yolov15-spci-2026.04',
  tenant_library_size: 0,
};

// ── tests ─────────────────────────────────────────────────────────────────────

describe('RecognitionPipeline', () => {
  let client: MLClient;
  let pipeline: RecognitionPipeline;

  beforeEach(() => {
    client = makeMockClient();
    pipeline = new RecognitionPipeline(client);
  });

  it('returns plan_id and page_number from input', async () => {
    const result = await pipeline.run(BASE_INPUT);
    expect(result.plan_id).toBe('plan-1');
    expect(result.page_number).toBe(1);
  });

  it('calls detectScale with correct args', async () => {
    await pipeline.run(BASE_INPUT);
    expect(client.detectScale).toHaveBeenCalledWith('plan-1', 1, '/tmp/test.png', 'tenant-1');
  });

  it('returns scale and scale_confidence from scale detection', async () => {
    const result = await pipeline.run(BASE_INPUT);
    expect(result.scale).toBe('1:100');
    expect(result.scale_confidence).toBe(0.92);
  });

  it('calls runYOLO once', async () => {
    await pipeline.run(BASE_INPUT);
    expect(client.runYOLO).toHaveBeenCalledTimes(1);
  });

  it('skips DINOv2 when tenant_library_size is 0', async () => {
    await pipeline.run({ ...BASE_INPUT, tenant_library_size: 0 });
    expect(client.matchDINOv2).not.toHaveBeenCalled();
  });

  it('calls DINOv2 when tenant_library_size > 0', async () => {
    (client.matchDINOv2 as ReturnType<typeof vi.fn>).mockResolvedValue(makeDINOv2Result());
    await pipeline.run({ ...BASE_INPUT, tenant_library_size: 5 });
    expect(client.matchDINOv2).toHaveBeenCalledTimes(1);
  });

  it('detections count matches raw YOLO output (no NMS overlap)', async () => {
    const result = await pipeline.run(BASE_INPUT);
    expect(result.stats.raw_yolo).toBe(2);
    expect(result.stats.after_nms).toBe(2);
  });

  it('all detections have a router_decision', async () => {
    const result = await pipeline.run(BASE_INPUT);
    for (const d of result.detections) {
      expect(['auto_accept', 'verify_with_vision', 'send_to_review', 'reject']).toContain(d.router_decision);
    }
  });

  it('duration_ms is positive', async () => {
    const result = await pipeline.run(BASE_INPUT);
    expect(result.duration_ms).toBeGreaterThan(0);
  });

  it('uses fallback scale when detectScale throws', async () => {
    (client.detectScale as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('timeout'));
    const result = await pipeline.run(BASE_INPUT);
    expect(result.scale).toBe('1:100');
    expect(result.scale_confidence).toBe(0.30);
    expect(result.warnings.some((w) => w.startsWith('scale_detection_failed'))).toBe(true);
  });

  it('collects YOLO warnings into result warnings', async () => {
    (client.runYOLO as ReturnType<typeof vi.fn>).mockResolvedValue(
      makeYOLOResult({ warnings: ['low_contrast_page'] }),
    );
    const result = await pipeline.run(BASE_INPUT);
    expect(result.warnings).toContain('low_contrast_page');
  });

  it('handles DINOv2 failure gracefully without crashing', async () => {
    (client.matchDINOv2 as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('dinov2 down'));
    const result = await pipeline.run({ ...BASE_INPUT, tenant_library_size: 5 });
    expect(result.detections.length).toBeGreaterThanOrEqual(0);
    expect(result.warnings.some((w) => w.includes('dinov2_failed'))).toBe(true);
  });

  it('propagates DINOv2 match into detection sources', async () => {
    (client.matchDINOv2 as ReturnType<typeof vi.fn>).mockResolvedValue({
      tenant_id: 'tenant-1',
      library_size: 10,
      matches: [
        {
          detection_id: 'det-001',
          matched: true,
          matched_class_slug: 'sprinkler-pendent-k57',
          similarity: 0.94,
          tier: 'high',
          agreement: 'agree',
        },
      ],
    });

    const result = await pipeline.run({ ...BASE_INPUT, tenant_library_size: 10 });
    const det = result.detections.find((d) => d.detection_id === 'det-001');
    expect(det).toBeDefined();
    expect(det!.sources.dinov2).toBeDefined();
    expect(det!.sources.dinov2!.confidence).toBeCloseTo(0.94);
  });

  it('returns correct model_version from YOLO result', async () => {
    const result = await pipeline.run(BASE_INPUT);
    expect(result.model_version).toBe('yolov15-spci-2026.04');
  });

  it('returns empty detections when YOLO finds nothing', async () => {
    (client.runYOLO as ReturnType<typeof vi.fn>).mockResolvedValue(
      makeYOLOResult({ total_detections: 0, tile_results: [
        {
          tile: { tile_id: 'tile-001', origin_x: 0, origin_y: 0, width: 1024, height: 1024 },
          detections: [],
        },
      ] }),
    );
    const result = await pipeline.run(BASE_INPUT);
    expect(result.detections).toHaveLength(0);
    expect(result.stats.raw_yolo).toBe(0);
  });
});
