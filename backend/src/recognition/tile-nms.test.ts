import { describe, it, expect } from 'vitest';
import { TileNMS } from './tile-nms.js';
import type { TileLayout } from '../types/index.js';

const layout: TileLayout[] = [
  { tile_id: 'tile-001', origin: [0, 0], size: [1024, 1024] },
  { tile_id: 'tile-002', origin: [819, 0], size: [1024, 1024] },
];

const makeDetection = (
  id: string,
  tile_id: string,
  bbox: { x1: number; y1: number; x2: number; y2: number },
  confidence = 0.85,
) => ({
  detection_id: id,
  tile_id,
  plan_id: 'plan-001',
  class_slug: 'sprinkler-pendent-k57',
  bbox,
  sources: { yolo: { confidence, model_version: 'yolov15-spci-2026.04' } },
  border_detection: false,
  tile_origin: layout.find((t) => t.tile_id === tile_id)!.origin,
  tile_size: layout.find((t) => t.tile_id === tile_id)!.size,
});

describe('TileNMS', () => {
  const nms = new TileNMS();

  it('keeps single detection unchanged', () => {
    const d = makeDetection('d1', 'tile-001', { x1: 100, y1: 100, x2: 135, y2: 135 });
    const result = nms.process([d], layout);
    expect(result.detections).toHaveLength(1);
    expect(result.stats.dropped).toBe(0);
  });

  it('deduplicates same symbol detected in two overlapping tiles', () => {
    // Symbol at page position ~860,100 — falls in overlap zone
    // tile-001 origin [0,0]: bbox [860, 100, 895, 135]
    // tile-002 origin [819,0]: bbox [41, 100, 76, 135]
    const d1 = makeDetection('d1', 'tile-001', { x1: 860, y1: 100, x2: 895, y2: 135 }, 0.88);
    const d2 = makeDetection('d2', 'tile-002', { x1: 41, y1: 100, x2: 76, y2: 135 }, 0.82);

    const result = nms.process([d1, d2], layout);

    expect(result.detections).toHaveLength(1);
    expect(result.stats.dropped).toBe(1);
    // Keeps highest confidence (d1 = 0.88)
    expect(result.detections[0].detection_id).toBe('d1');
    expect(result.detections[0].merged_from).toContain('d2');
  });

  it('keeps two distinct detections in different regions', () => {
    const d1 = makeDetection('d1', 'tile-001', { x1: 100, y1: 100, x2: 135, y2: 135 });
    const d2 = makeDetection('d2', 'tile-002', { x1: 400, y1: 400, x2: 435, y2: 435 });

    const result = nms.process([d1, d2], layout);
    expect(result.detections).toHaveLength(2);
    expect(result.stats.dropped).toBe(0);
  });

  it('flags border detections correctly', () => {
    // bbox touching right edge of tile-001 (x2 > 1024-5 = 1019)
    const d = makeDetection('d1', 'tile-001', { x1: 990, y1: 100, x2: 1024, y2: 135 });
    const result = nms.process([d], layout);
    expect(result.detections[0].border_detection).toBe(true);
    expect(result.stats.border_flagged).toBe(1);
  });

  it('converts tile coordinates to page coordinates correctly', () => {
    // tile-002 origin [819, 0], bbox [100, 50, 135, 85]
    // page bbox should be [919, 50, 954, 85]
    const d = makeDetection('d1', 'tile-002', { x1: 100, y1: 50, x2: 135, y2: 85 });
    const result = nms.process([d], layout);
    const bbox = result.detections[0].bbox_page;
    expect(bbox.x1).toBe(919);
    expect(bbox.y1).toBe(50);
    expect(bbox.x2).toBe(954);
    expect(bbox.y2).toBe(85);
  });

  it('reports correct stats', () => {
    // d1 and d2 overlap in page coords (both map to ~[860,100,895,135])
    const detections = [
      makeDetection('d1', 'tile-001', { x1: 860, y1: 100, x2: 895, y2: 135 }, 0.88),
      makeDetection('d2', 'tile-002', { x1: 41, y1: 100, x2: 76, y2: 135 }, 0.82),
      makeDetection('d3', 'tile-001', { x1: 300, y1: 300, x2: 335, y2: 335 }),
    ];
    const result = nms.process(detections, layout);
    expect(result.stats.raw_count).toBe(3);
    expect(result.stats.after_nms_count).toBe(2);
    expect(result.stats.dropped).toBe(1);
  });
});
