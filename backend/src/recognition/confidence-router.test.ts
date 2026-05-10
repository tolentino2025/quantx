import { describe, it, expect, beforeEach } from 'vitest';
import { ConfidenceRouter } from './confidence-router.js';
import type { RawDetection } from '../types/index.js';
import { resolve } from 'path';

const CONFIG_PATH = resolve(__dirname, '../../../config/confidence-thresholds.json');

const makeDetection = (overrides: Partial<RawDetection> = {}): RawDetection => ({
  detection_id: 'test-uuid',
  tile_id: 'tile-001',
  plan_id: 'plan-001',
  class_slug: 'sprinkler-pendent-k57',
  bbox: { x1: 100, y1: 100, x2: 135, y2: 135 },
  sources: { yolo: { confidence: 0.90, model_version: 'yolov15-spci-2026.04' } },
  border_detection: false,
  ...overrides,
});

describe('ConfidenceRouter', () => {
  let router: ConfidenceRouter;

  beforeEach(() => {
    router = new ConfidenceRouter(CONFIG_PATH);
    router.resetBudget();
  });

  it('auto_accept when YOLO confidence above threshold', () => {
    const result = router.route(makeDetection({ sources: { yolo: { confidence: 0.92 } } }), 50);
    expect(result.router_decision).toBe('auto_accept');
  });

  it('verify_with_vision when confidence in middle range', () => {
    const result = router.route(makeDetection({ sources: { yolo: { confidence: 0.75 } } }), 50);
    expect(result.router_decision).toBe('verify_with_vision');
  });

  it('send_to_review when confidence below verify threshold', () => {
    const result = router.route(makeDetection({ sources: { yolo: { confidence: 0.55 } } }), 50);
    expect(result.router_decision).toBe('send_to_review');
  });

  it('reject when confidence below floor', () => {
    const result = router.route(makeDetection({ sources: { yolo: { confidence: 0.30 } } }), 50);
    expect(result.router_decision).toBe('reject');
  });

  it('boosts confidence when YOLO and DINOv2 agree', () => {
    const detection = makeDetection({
      sources: {
        yolo: { confidence: 0.80 },
        dinov2: { confidence: 0.82, top1_slug: 'sprinkler-pendent-k57' },
      },
    });
    const result = router.route(detection, 50);
    expect(result.confidence_final).toBeGreaterThan(0.85);
    expect(result.source_agreement).toBe('agree');
    expect(result.router_decision).toBe('auto_accept');
  });

  it('caps confidence when YOLO and DINOv2 disagree', () => {
    const detection = makeDetection({
      sources: {
        yolo: { confidence: 0.90 },
        dinov2: { confidence: 0.88, top1_slug: 'sprinkler-upright-k57' },
      },
    });
    const result = router.route(detection, 50);
    expect(result.confidence_final).toBeLessThanOrEqual(0.65);
    expect(result.source_agreement).toBe('disagree');
  });

  it('downgrades decision by one level for border detections', () => {
    const detection = makeDetection({
      sources: { yolo: { confidence: 0.92 } },
      border_detection: true,
    });
    const result = router.route(detection, 50);
    expect(result.router_decision).toBe('verify_with_vision');
  });

  it('sends to review when budget is insufficient for vision', () => {
    const detection = makeDetection({ sources: { yolo: { confidence: 0.75 } } });
    const result = router.route(detection, 1.0);
    expect(result.router_decision).toBe('send_to_review');
    expect(result.router_justification).toContain('Budget insuficiente');
  });

  it('includes justification with threshold source', () => {
    const result = router.route(makeDetection(), 50);
    expect(result.router_justification).toContain('sprinkler-pendent-k57');
    expect(result.router_justification).toContain('auto_accept');
  });

  it('uses defaults for unknown class', () => {
    const detection = makeDetection({ class_slug: 'classe-desconhecida-xyz' });
    const result = router.route(detection, 50);
    expect(result.router_justification).toContain('defaults');
  });

  it('processes batch and accumulates budget', () => {
    const detections = Array.from({ length: 5 }, (_, i) =>
      makeDetection({
        detection_id: `uuid-${i}`,
        sources: { yolo: { confidence: 0.75 } },
      }),
    );
    const results = router.routeBatch(detections, 50);
    expect(results).toHaveLength(5);
    const visionCalls = results.filter((r) => r.router_decision === 'verify_with_vision');
    expect(router.getBudgetUsed()).toBe(visionCalls.length * 2.5);
  });
});
