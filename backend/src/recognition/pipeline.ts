import { appendFileSync, mkdirSync } from 'fs';
import { join } from 'path';
import { randomUUID } from 'crypto';
import type { RawDetection, RoutedDetection, TileLayout } from '../types/index.js';
import { RecognitionOrchestrator } from './orchestrator.js';
import { ConfidenceRouter } from './confidence-router.js';
import { TileNMS } from './tile-nms.js';
import type {
  MLClient,
  YOLODetection,
  YOLOTileResult,
  DINOv2MatchEntry,
  ScaleDetectionResponse,
} from '../workers/ml-client.js';

export interface PipelineInput {
  plan_id: string;
  page_number: number;
  tenant_id: string;
  image_path: string;
  model_version: string;
  tenant_library_size: number;
  audit_mode?: boolean;
  budget_cents?: number;
}

export interface PipelineResult {
  plan_id: string;
  page_number: number;
  model_version: string;
  scale: string;
  scale_confidence: number;
  detections: RoutedDetection[];
  stats: {
    raw_yolo: number;
    after_nms: number;
    auto_accepted: number;
    verify_with_vision: number;
    send_to_review: number;
    rejected: number;
    vision_cost_cents: number;
  };
  warnings: string[];
  duration_ms: number;
}

const TELEMETRY_DIR = '/tmp/quantx-telemetry';

export class RecognitionPipeline {
  private readonly orchestrator: RecognitionOrchestrator;
  private readonly router: ConfidenceRouter;
  private readonly nms: TileNMS;
  private readonly mlClient: MLClient;

  constructor(mlClient: MLClient) {
    this.orchestrator = new RecognitionOrchestrator();
    this.router = new ConfidenceRouter();
    this.nms = new TileNMS();
    this.mlClient = mlClient;
  }

  async run(input: PipelineInput): Promise<PipelineResult> {
    const startMs = Date.now();
    const warnings: string[] = [];
    const budget = input.budget_cents ?? 50.0;

    // ── Step 1: Scale detection ────────────────────────────────────────────
    const scaleResult = await this._detectScale(input, warnings);
    const { scale, scale_confidence, recommended, legend_region } = scaleResult;

    // ── Step 2: Orchestrator plan ──────────────────────────────────────────
    const plan = this.orchestrator.plan({
      plan_id: input.plan_id,
      page_number: input.page_number,
      tenant_id: input.tenant_id,
      tile_list: [],            // filled by YOLO engine
      tenant_library_size: input.tenant_library_size,
      scale,
      scale_confidence,
      audit_mode: input.audit_mode ?? false,
      budget_remaining_cents: budget,
    });

    warnings.push(...plan.warnings);

    // ── Step 3: YOLO + DINOv2 in parallel ─────────────────────────────────
    const [yoloResult, dinov2Result] = await Promise.all([
      this._runYOLO(input, recommended, legend_region, warnings),
      input.tenant_library_size > 0
        ? this._runDINOv2(input, [], recommended.dinov2_window, warnings)
            .catch((err) => { warnings.push(`dinov2_failed: ${err.message}`); return null; })
        : Promise.resolve(null),
    ]);

    const allYoloDetections = yoloResult.tile_results.flatMap((tr) => tr.detections);

    // If DINOv2 was deferred (needed flat list first), run it now
    const dinov2Matches = dinov2Result
      ?? await (async () => {
        if (input.tenant_library_size === 0 || allYoloDetections.length === 0) return null;
        return this._runDINOv2(input, allYoloDetections, recommended.dinov2_window, warnings)
          .catch((err) => { warnings.push(`dinov2_failed: ${err.message}`); return null; });
      })();

    // ── Step 4: Merge YOLO + DINOv2 into unified detection candidates ──────
    const dinov2Map = this._buildDINOv2Map(dinov2Matches?.matches ?? []);
    const tileLayout = this._extractTileLayout(yoloResult.tile_results);
    const rawDetections = this._mergeDetections(allYoloDetections, dinov2Map);

    // ── Step 5: Cross-tile NMS ─────────────────────────────────────────────
    const nmsResult = this.nms.process(
      rawDetections.map((d) => ({
        ...d,
        tile_origin: (() => {
          const tile = tileLayout.find((t) => t.tile_id === d.tile_id);
          return [tile?.origin[0] ?? 0, tile?.origin[1] ?? 0] as [number, number];
        })(),
        tile_size: (() => {
          const tile = tileLayout.find((t) => t.tile_id === d.tile_id);
          return [tile?.size[0] ?? 1024, tile?.size[1] ?? 1024] as [number, number];
        })(),
      })),
      tileLayout,
    );

    // ── Step 6: Confidence routing ─────────────────────────────────────────
    this.router.resetBudget();
    let remaining = budget;
    const routed = this.router.routeBatch(
      nmsResult.detections.map((d) => ({
        detection_id: d.detection_id,
        tile_id: d.tile_id,
        plan_id: input.plan_id,
        class_slug: d.class_slug,
        bbox: d.bbox,
        sources: d.sources,
        border_detection: d.border_detection,
      })),
      remaining,
    );

    remaining -= this.router.getBudgetUsed();

    // ── Step 7: Audit trail ────────────────────────────────────────────────
    this._writeTelemetry(input, routed, scale, yoloResult.model_version);

    const stats = this._computeStats(routed, allYoloDetections.length);

    return {
      plan_id: input.plan_id,
      page_number: input.page_number,
      model_version: yoloResult.model_version,
      scale,
      scale_confidence,
      detections: routed,
      stats,
      warnings,
      duration_ms: Date.now() - startMs,
    };
  }

  private async _detectScale(
    input: PipelineInput,
    warnings: string[],
  ): Promise<ScaleDetectionResponse> {
    try {
      return await this.mlClient.detectScale(
        input.plan_id,
        input.page_number,
        input.image_path,
        input.tenant_id,
      );
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      warnings.push(`scale_detection_failed: ${msg} — using fallback 1:100`);
      return {
        plan_id: input.plan_id,
        page_number: input.page_number,
        scale: '1:100',
        scale_confidence: 0.30,
        method: 'fallback',
        cartouche_region: null,
        legend_region: null,
        recommended: { dpi: 300, tile_size: 1024, overlap: 0.20, imgsz: 1024, dinov2_window: 128 },
        uncertainty_note: 'Scale detection failed — using fallback 1:100',
        warnings: ['scale_detection_failed'],
      };
    }
  }

  private async _runYOLO(
    input: PipelineInput,
    recommended: ScaleDetectionResponse['recommended'],
    legendRegion: ScaleDetectionResponse['legend_region'],
    warnings: string[],
  ) {
    const result = await this.mlClient.runYOLO({
      planId: input.plan_id,
      pageNumber: input.page_number,
      imagePath: input.image_path,
      modelVersion: input.model_version,
      tileSize: recommended.tile_size,
      overlap: recommended.overlap,
      imgsz: recommended.imgsz,
      legendRegion,
    });
    warnings.push(...result.warnings);
    return result;
  }

  private async _runDINOv2(
    input: PipelineInput,
    detections: YOLODetection[],
    dinov2Window: number,
    warnings: string[],
  ) {
    const result = await this.mlClient.matchDINOv2({
      tenantId: input.tenant_id,
      imagePath: input.image_path,
      detections,
      dinov2Window,
    });
    if (result.warning) warnings.push(result.warning);
    return result;
  }

  private _buildDINOv2Map(matches: DINOv2MatchEntry[]): Map<string, DINOv2MatchEntry> {
    return new Map(matches.map((m) => [m.detection_id, m]));
  }

  private _extractTileLayout(tileResults: YOLOTileResult[]): TileLayout[] {
    return tileResults.map((tr) => ({
      tile_id: tr.tile.tile_id,
      origin: [tr.tile.origin_x, tr.tile.origin_y] as [number, number],
      size: [tr.tile.width, tr.tile.height] as [number, number],
    }));
  }

  private _mergeDetections(
    yoloDetections: YOLODetection[],
    dinov2Map: Map<string, DINOv2MatchEntry>,
  ): RawDetection[] {
    return yoloDetections.map((d) => {
      const match = dinov2Map.get(d.detection_id);
      return {
        detection_id: d.detection_id,
        tile_id: d.tile_id,
        plan_id: '',
        class_slug: d.class_slug,
        bbox: d.bbox_tile,
        sources: {
          yolo: { confidence: d.confidence, model_version: d.model_version },
          ...(match?.matched
            ? {
                dinov2: {
                  confidence: match.similarity,
                  top1_slug: match.matched_class_slug,
                },
              }
            : {}),
        },
        border_detection: false,
      };
    });
  }

  private _computeStats(detections: RoutedDetection[], rawYoloCount: number) {
    const byDecision = (d: string) => detections.filter((r) => r.router_decision === d).length;
    return {
      raw_yolo: rawYoloCount,
      after_nms: detections.length,
      auto_accepted: byDecision('auto_accept'),
      verify_with_vision: byDecision('verify_with_vision'),
      send_to_review: byDecision('send_to_review'),
      rejected: byDecision('reject'),
      vision_cost_cents: this.router.getBudgetUsed(),
    };
  }

  private _writeTelemetry(
    input: PipelineInput,
    detections: RoutedDetection[],
    scale: string,
    modelVersion: string,
  ): void {
    try {
      mkdirSync(TELEMETRY_DIR, { recursive: true });
      const date = new Date().toISOString().slice(0, 10).replace(/-/g, '');
      const logPath = join(TELEMETRY_DIR, `${date}_recognition.jsonl`);
      const entry = {
        timestamp: new Date().toISOString(),
        plan_id: input.plan_id,
        page_number: input.page_number,
        tenant_id: input.tenant_id,
        scale,
        model_version: modelVersion,
        detection_count: detections.length,
        decisions: {
          auto_accept: detections.filter((d) => d.router_decision === 'auto_accept').length,
          verify_with_vision: detections.filter((d) => d.router_decision === 'verify_with_vision').length,
          send_to_review: detections.filter((d) => d.router_decision === 'send_to_review').length,
          rejected: detections.filter((d) => d.router_decision === 'reject').length,
        },
      };
      appendFileSync(logPath, JSON.stringify(entry) + '\n');
    } catch {
      // telemetry failure must never crash the pipeline
    }
  }
}
