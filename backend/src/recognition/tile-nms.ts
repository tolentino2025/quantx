import type { RawDetection, BBox, TileLayout } from '../types/index.js';

type NmsAction = 'kept' | 'merged' | 'dropped';

export interface NmsDetection extends RawDetection {
  bbox_page: BBox;
  nms_action: NmsAction;
  merged_from: string[];
}

export interface NmsResult {
  detections: NmsDetection[];
  stats: {
    raw_count: number;
    after_nms_count: number;
    dropped: number;
    merged: number;
    border_flagged: number;
  };
}

interface TileDetection extends RawDetection {
  tile_origin: [number, number];
  tile_size: [number, number];
}

const BORDER_MARGIN_PX = 5;
const IOU_THRESHOLD = 0.45;

export class TileNMS {
  process(
    rawDetections: TileDetection[],
    tileLayout: TileLayout[],
  ): NmsResult {
    const layoutMap = new Map(tileLayout.map((t) => [t.tile_id, t]));

    // Convert all bboxes to page coordinates
    const paged = rawDetections.map((d) => {
      const tile = layoutMap.get(d.tile_id);
      const origin = tile?.origin ?? [0, 0];
      return {
        ...d,
        bbox_page: this.toPageCoords(d.bbox, origin),
        border_detection: tile ? this.isBorderDetection(d.bbox, tile.size) : false,
        nms_action: 'kept' as NmsAction,
        merged_from: [d.detection_id],
      };
    });

    const deduplicated = this.applyNMS(paged);
    const borderFlagged = deduplicated.filter((d) => d.border_detection).length;
    const dropped = paged.length - deduplicated.length;

    return {
      detections: deduplicated,
      stats: {
        raw_count: rawDetections.length,
        after_nms_count: deduplicated.length,
        dropped,
        merged: deduplicated.filter((d) => d.merged_from.length > 1).length,
        border_flagged: borderFlagged,
      },
    };
  }

  private applyNMS(detections: NmsDetection[]): NmsDetection[] {
    const result: NmsDetection[] = [];
    const dropped = new Set<string>();

    for (let i = 0; i < detections.length; i++) {
      if (dropped.has(detections[i].detection_id)) continue;

      let current = { ...detections[i] };

      for (let j = i + 1; j < detections.length; j++) {
        if (dropped.has(detections[j].detection_id)) continue;
        if (detections[i].tile_id === detections[j].tile_id) continue;

        const iou = this.computeIoU(current.bbox_page, detections[j].bbox_page);

        if (iou >= IOU_THRESHOLD) {
          if (current.class_slug === detections[j].class_slug) {
            // Same class: keep highest confidence, drop other
            const currentConf = this.maxConfidence(current);
            const otherConf = this.maxConfidence(detections[j]);

            if (currentConf >= otherConf) {
              dropped.add(detections[j].detection_id);
              current.merged_from.push(...detections[j].merged_from);
              current.nms_action = 'merged';
            } else {
              dropped.add(current.detection_id);
              current = {
                ...detections[j],
                merged_from: [...detections[j].merged_from, ...current.merged_from],
                nms_action: 'merged',
              };
            }
          }
          // Different classes with high IoU: keep both, flagged for review
          // (handled by not dropping either)
        } else {
          const centroidDist = this.centroidDistance(current.bbox_page, detections[j].bbox_page);
          const symbolSize = this.estimateSymbolSize(current.bbox_page);

          if (centroidDist < symbolSize * 0.5) {
            // Close centroids but low IoU: merge bboxes
            current = {
              ...current,
              bbox_page: this.unionBBox(current.bbox_page, detections[j].bbox_page),
              merged_from: [...current.merged_from, ...detections[j].merged_from],
              nms_action: 'merged',
            };
            dropped.add(detections[j].detection_id);
          }
        }
      }

      result.push(current);
    }

    return result;
  }

  private toPageCoords(bbox: BBox, origin: [number, number]): BBox {
    return {
      x1: origin[0] + bbox.x1,
      y1: origin[1] + bbox.y1,
      x2: origin[0] + bbox.x2,
      y2: origin[1] + bbox.y2,
    };
  }

  private isBorderDetection(bbox: BBox, tileSize: [number, number]): boolean {
    return (
      bbox.x1 < BORDER_MARGIN_PX ||
      bbox.y1 < BORDER_MARGIN_PX ||
      bbox.x2 > tileSize[0] - BORDER_MARGIN_PX ||
      bbox.y2 > tileSize[1] - BORDER_MARGIN_PX
    );
  }

  private computeIoU(a: BBox, b: BBox): number {
    const ix1 = Math.max(a.x1, b.x1);
    const iy1 = Math.max(a.y1, b.y1);
    const ix2 = Math.min(a.x2, b.x2);
    const iy2 = Math.min(a.y2, b.y2);

    if (ix2 <= ix1 || iy2 <= iy1) return 0;

    const intersection = (ix2 - ix1) * (iy2 - iy1);
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);

    return intersection / (areaA + areaB - intersection);
  }

  private centroidDistance(a: BBox, b: BBox): number {
    const cax = (a.x1 + a.x2) / 2;
    const cay = (a.y1 + a.y2) / 2;
    const cbx = (b.x1 + b.x2) / 2;
    const cby = (b.y1 + b.y2) / 2;
    return Math.sqrt((cax - cbx) ** 2 + (cay - cby) ** 2);
  }

  private estimateSymbolSize(bbox: BBox): number {
    return Math.max(bbox.x2 - bbox.x1, bbox.y2 - bbox.y1);
  }

  private unionBBox(a: BBox, b: BBox): BBox {
    return {
      x1: Math.min(a.x1, b.x1),
      y1: Math.min(a.y1, b.y1),
      x2: Math.max(a.x2, b.x2),
      y2: Math.max(a.y2, b.y2),
    };
  }

  private maxConfidence(d: RawDetection): number {
    return Math.max(
      d.sources.yolo?.confidence ?? 0,
      d.sources.dinov2?.confidence ?? 0,
      d.sources.rtdetr?.confidence ?? 0,
    );
  }
}
