export type DetectionSource = 'yolo' | 'dinov2' | 'rtdetr' | 'vision' | 'template';

export type RouterDecision =
  | 'auto_accept'
  | 'verify_with_vision'
  | 'send_to_review'
  | 'reject'
  | 'fallback';

export type SourceAgreement = 'agree' | 'disagree' | 'single_source';

export type CorrectionAction =
  | 'accept'
  | 'reject'
  | 'reclassify'
  | 'adjust_bbox'
  | 'add_new';

export interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface SourceResult {
  confidence: number;
  model_version?: string;
  top1_slug?: string;
}

export interface RawDetection {
  detection_id: string;
  tile_id: string;
  plan_id: string;
  class_slug: string;
  bbox: BBox;
  sources: Partial<Record<DetectionSource, SourceResult>>;
  border_detection: boolean;
}

export interface RoutedDetection extends RawDetection {
  confidence_final: number;
  source_agreement: SourceAgreement;
  router_decision: RouterDecision;
  router_justification: string;
  cost_cents: number;
  timestamp: string;
}

export interface ClassThreshold {
  auto_accept: number;
  verify_with_vision: number;
  reject: number;
  calibrated: boolean;
  last_calibration?: string;
  sample_size?: number;
  notes?: string;
}

export interface ThresholdConfig {
  defaults: ClassThreshold;
  budget: {
    hard_cap_cents: number;
    alert_cents: number;
    estimated_vision_cost_cents: number;
  };
  source_weights: Record<DetectionSource, { weight: number }>;
  agreement_rules: {
    yolo_dinov2_agree_boost: number;
    yolo_dinov2_disagree_cap: number;
    border_detection_penalty: string;
    uncalibrated_max_auto_accept: number;
  };
  by_class: Record<string, ClassThreshold>;
}

export interface OrchestratorInput {
  plan_id: string;
  page_number: number;
  tenant_id: string;
  tile_list: string[];
  tenant_library_size: number;
  scale: string;
  scale_confidence: number;
  audit_mode: boolean;
  budget_remaining_cents: number;
}

export interface PipelineStep {
  pipeline: DetectionSource;
  tiles: string[];
  parallel: boolean;
  config: Record<string, unknown>;
}

export interface OrchestratorPlan {
  plan_id: string;
  pipeline_plan: PipelineStep[];
  expected_cost_cents: number;
  next_agent: string;
  justification: string;
  warnings: string[];
}

export interface ScaleConfig {
  dpi: number;
  tile_size: number;
  overlap: number;
  imgsz: number;
}

export interface TileLayout {
  tile_id: string;
  origin: [number, number];
  size: [number, number];
}
