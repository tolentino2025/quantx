import { readFileSync } from 'fs';
import { resolve } from 'path';
import type {
  OrchestratorInput,
  OrchestratorPlan,
  PipelineStep,
  ScaleConfig,
} from '../types/index.js';

interface PipelineRules {
  tile_config: {
    scales: Record<string, ScaleConfig & { note?: string }>;
  };
}

const RULES_PATH = resolve(process.cwd(), '../config/pipeline-rules.json');

const SCALE_DEFAULTS: ScaleConfig = { dpi: 300, tile_size: 1024, overlap: 0.20, imgsz: 1024 };

export class RecognitionOrchestrator {
  private rules: PipelineRules;

  constructor(rulesPath = RULES_PATH) {
    this.rules = JSON.parse(readFileSync(rulesPath, 'utf-8')) as PipelineRules;
  }

  plan(input: OrchestratorInput): OrchestratorPlan {
    const warnings: string[] = [];
    this.validateInput(input, warnings);

    const scaleConfig = this.resolveScaleConfig(input.scale, warnings);
    const pipeline_plan: PipelineStep[] = [];
    const justification_parts: string[] = [];

    // YOLO — sempre primeiro em todos os tiles
    pipeline_plan.push({
      pipeline: 'yolo',
      tiles: [...input.tile_list],
      parallel: true,
      config: {
        imgsz: scaleConfig.imgsz,
        conf: 0.25,
        iou: 0.45,
        tile_size: scaleConfig.tile_size,
        overlap: scaleConfig.overlap,
        use_sahi: true,
      },
    });
    justification_parts.push('YOLO rodando em todos os tiles como passada base');

    // DINOv2 — em paralelo se o tenant tem biblioteca pessoal
    if (input.tenant_library_size > 0) {
      pipeline_plan.push({
        pipeline: 'dinov2',
        tiles: [...input.tile_list],
        parallel: true,
        config: {
          top_k: 5,
          threshold: 0.75,
          dinov2_window: this.resolveDINOv2Window(input.scale),
          tenant_id: 'resolve_from_context',
        },
      });
      justification_parts.push(
        `DINOv2 em paralelo (biblioteca do tenant: ${input.tenant_library_size} classe(s))`,
      );
    } else {
      justification_parts.push('DINOv2 ignorado (biblioteca pessoal vazia)');
    }

    // Template Matching — apenas em audit_mode
    if (input.audit_mode) {
      pipeline_plan.push({
        pipeline: 'template',
        tiles: [...input.tile_list],
        parallel: false,
        config: { audit_mode: true },
      });
      justification_parts.push('Template Matching incluído (audit_mode=true)');
      warnings.push('audit_mode ativo: Template Matching habilitado');
    }

    // Alerta de scale_confidence baixo
    if (input.scale_confidence < 0.70) {
      warnings.push(
        `scale_confidence=${input.scale_confidence.toFixed(2)} abaixo de 0.70 — escala incerta, tile config pode ser inadequado`,
      );
    }

    // Alerta de budget baixo
    if (input.budget_remaining_cents < 5.0) {
      warnings.push(
        `budget_remaining=$${input.budget_remaining_cents.toFixed(2)}¢ — muito baixo, Vision API será bloqueada pelo ConfidenceRouter`,
      );
    }

    return {
      plan_id: input.plan_id,
      pipeline_plan,
      expected_cost_cents: 0,
      next_agent: 'tile-nms → confidence-router',
      justification: justification_parts.join('. ') + '.',
      warnings,
    };
  }

  private validateInput(input: OrchestratorInput, warnings: string[]): void {
    if (!input.plan_id) throw new Error('plan_id é obrigatório');
    if (!input.tenant_id) throw new Error('tenant_id é obrigatório');
    if (!input.tile_list?.length) throw new Error('tile_list não pode ser vazio');

    if (input.scale_confidence === null || input.scale_confidence === undefined) {
      throw new Error(
        'scale_confidence ausente — execute Scale Detection Agent antes do Orchestrator',
      );
    }
  }

  private resolveScaleConfig(scale: string, warnings: string[]): ScaleConfig {
    const scales = this.rules.tile_config.scales;
    const config = scales[scale];

    if (!config) {
      warnings.push(`Escala '${scale}' não mapeada — usando config de '1:100' como fallback`);
      return scales['1:100'] ?? SCALE_DEFAULTS;
    }

    if (config.note) warnings.push(config.note);
    return config;
  }

  private resolveDINOv2Window(scale: string): number {
    const windows: Record<string, number> = {
      '1:50': 96,
      '1:75': 112,
      '1:100': 128,
      '1:150': 128,
      '1:200': 160,
      '1:250': 160,
    };
    return windows[scale] ?? 128;
  }
}
