import type { BBox, ScaleConfig } from '../types/index.js';

export interface ScaleDetectionResponse {
  plan_id: string;
  page_number: number;
  scale: string;
  scale_confidence: number;
  method: string;
  cartouche_region: BBox | null;
  legend_region: BBox | null;
  recommended: ScaleConfig & { dinov2_window: number };
  uncertainty_note: string;
  warnings: string[];
}

export interface YOLODetection {
  detection_id: string;
  tile_id: string;
  class_slug: string;
  bbox_tile: BBox;
  confidence: number;
  source: 'yolo';
  model_version: string;
}

export interface YOLOTileResult {
  tile: {
    tile_id: string;
    origin_x: number;
    origin_y: number;
    width: number;
    height: number;
  };
  detections: YOLODetection[];
}

export interface YOLOInferenceResponse {
  plan_id: string;
  page_number: number;
  model_version: string;
  total_detections: number;
  warnings: string[];
  tile_results: YOLOTileResult[];
}

export interface DINOv2MatchEntry {
  detection_id: string;
  matched: boolean;
  matched_class_slug?: string;
  yolo_class_slug?: string;
  similarity: number;
  tier: 'high' | 'medium' | 'low' | 'no_match';
  agreement?: 'agree' | 'disagree';
  top_k?: Array<{ class_slug: string; similarity: number }>;
}

export interface DINOv2MatchResponse {
  tenant_id: string;
  library_size: number;
  matches: DINOv2MatchEntry[];
  warning?: string;
}

export interface MLClientOptions {
  baseUrl: string;
  timeoutMs?: number;
  retries?: number;
}

export class MLClientError extends Error {
  constructor(
    message: string,
    public readonly status?: number,
    public readonly endpoint?: string,
  ) {
    super(message);
    this.name = 'MLClientError';
  }
}

export class MLClient {
  private readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly retries: number;

  constructor({ baseUrl, timeoutMs = 30_000, retries = 2 }: MLClientOptions) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.timeoutMs = timeoutMs;
    this.retries = retries;
  }

  async detectScale(
    planId: string,
    pageNumber: number,
    imagePath: string,
    tenantId: string,
  ): Promise<ScaleDetectionResponse> {
    return this._post<ScaleDetectionResponse>('/scale-detection', {
      plan_id: planId,
      page_number: pageNumber,
      image_path: imagePath,
      tenant_id: tenantId,
    });
  }

  async runYOLO(params: {
    planId: string;
    pageNumber: number;
    imagePath: string;
    modelVersion: string;
    tileSize: number;
    overlap: number;
    imgsz: number;
    conf?: number;
    iou?: number;
    legendRegion?: BBox | null;
  }): Promise<YOLOInferenceResponse> {
    return this._post<YOLOInferenceResponse>('/yolo-inference', {
      plan_id: params.planId,
      page_number: params.pageNumber,
      image_path: params.imagePath,
      model_version: params.modelVersion,
      tile_size: params.tileSize,
      overlap: params.overlap,
      imgsz: params.imgsz,
      conf: params.conf ?? 0.25,
      iou: params.iou ?? 0.45,
      legend_region: params.legendRegion ?? null,
    });
  }

  async matchDINOv2(params: {
    tenantId: string;
    imagePath: string;
    detections: YOLODetection[];
    dinov2Window: number;
    topK?: number;
  }): Promise<DINOv2MatchResponse> {
    return this._post<DINOv2MatchResponse>('/dinov2/match', {
      tenant_id: params.tenantId,
      image_path: params.imagePath,
      detections: params.detections,
      dinov2_window: params.dinov2Window,
      top_k: params.topK ?? 5,
    });
  }

  async health(): Promise<boolean> {
    try {
      const res = await fetch(`${this.baseUrl}/health`, {
        signal: AbortSignal.timeout(3_000),
      });
      return res.ok;
    } catch {
      return false;
    }
  }

  private async _post<T>(endpoint: string, body: unknown): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    let lastError: Error = new MLClientError('Unknown error');

    for (let attempt = 0; attempt <= this.retries; attempt++) {
      try {
        const res = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: AbortSignal.timeout(this.timeoutMs),
        });

        if (!res.ok) {
          const text = await res.text().catch(() => '');
          throw new MLClientError(
            `ML service returned ${res.status}: ${text.slice(0, 200)}`,
            res.status,
            endpoint,
          );
        }

        return (await res.json()) as T;
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));

        // Don't retry on 4xx (client errors)
        if (err instanceof MLClientError && err.status && err.status < 500) throw err;

        if (attempt < this.retries) {
          const backoffMs = 500 * 2 ** attempt;
          await new Promise((r) => setTimeout(r, backoffMs));
        }
      }
    }

    throw lastError;
  }
}
