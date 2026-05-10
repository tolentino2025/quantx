import { readFileSync } from 'fs';
import { resolve } from 'path';
import { randomUUID } from 'crypto';
import type {
  RawDetection,
  RoutedDetection,
  RouterDecision,
  SourceAgreement,
  ThresholdConfig,
  ClassThreshold,
} from '../types/index.js';

const CONFIG_PATH = resolve(process.cwd(), '../config/confidence-thresholds.json');

export class ConfidenceRouter {
  private config: ThresholdConfig;
  private budgetUsedCents = 0;

  constructor(configPath = CONFIG_PATH) {
    this.config = JSON.parse(readFileSync(configPath, 'utf-8')) as ThresholdConfig;
  }

  resetBudget(): void {
    this.budgetUsedCents = 0;
  }

  getBudgetUsed(): number {
    return this.budgetUsedCents;
  }

  route(detection: RawDetection, budgetRemainingCents: number): RoutedDetection {
    const thresholds = this.getThresholds(detection.class_slug);
    const { confidence_final, source_agreement, agreement_note } =
      this.computeFinalConfidence(detection, thresholds);
    const { decision, budget_note } = this.decide(
      confidence_final,
      thresholds,
      detection.border_detection,
      budgetRemainingCents,
    );

    const cost_cents = decision === 'verify_with_vision'
      ? this.config.budget.estimated_vision_cost_cents
      : 0;

    if (decision === 'verify_with_vision') {
      this.budgetUsedCents += cost_cents;
    }

    return {
      ...detection,
      detection_id: detection.detection_id || randomUUID(),
      confidence_final,
      source_agreement,
      router_decision: decision,
      router_justification: this.buildJustification(
        detection.class_slug,
        confidence_final,
        decision,
        thresholds,
        source_agreement,
        agreement_note,
        budget_note,
      ),
      cost_cents,
      timestamp: new Date().toISOString(),
    };
  }

  routeBatch(
    detections: RawDetection[],
    budgetRemainingCents: number,
  ): RoutedDetection[] {
    let remaining = budgetRemainingCents;
    return detections.map((d) => {
      const routed = this.route(d, remaining);
      remaining -= routed.cost_cents;
      return routed;
    });
  }

  private getThresholds(classSlug: string): ClassThreshold & { is_default: boolean } {
    const perClass = this.config.by_class[classSlug];
    if (perClass) return { ...perClass, is_default: false };
    return { ...this.config.defaults, is_default: true };
  }

  private computeFinalConfidence(
    detection: RawDetection,
    thresholds: ClassThreshold,
  ): { confidence_final: number; source_agreement: SourceAgreement; agreement_note: string } {
    const { sources } = detection;
    const { agreement_rules, source_weights } = this.config;

    const yolo = sources.yolo;
    const dinov2 = sources.dinov2;

    let raw_confidence: number;
    let source_agreement: SourceAgreement;
    let agreement_note: string;

    if (yolo && dinov2) {
      const yolo_weighted = yolo.confidence * (source_weights.yolo?.weight ?? 1.0);
      const dinov2_weighted = dinov2.confidence * (source_weights.dinov2?.weight ?? 0.95);
      const avg = (yolo_weighted + dinov2_weighted) / 2;

      const classes_match = dinov2.top1_slug === detection.class_slug;

      if (classes_match) {
        source_agreement = 'agree';
        raw_confidence = Math.min(0.99, avg + agreement_rules.yolo_dinov2_agree_boost);
        agreement_note = `YOLO ${yolo.confidence.toFixed(3)} + DINOv2 ${dinov2.confidence.toFixed(3)} concordam (+${agreement_rules.yolo_dinov2_agree_boost} boost)`;
      } else {
        source_agreement = 'disagree';
        raw_confidence = Math.min(avg, agreement_rules.yolo_dinov2_disagree_cap);
        agreement_note = `YOLO aponta "${detection.class_slug}", DINOv2 aponta "${dinov2.top1_slug}" (cap em ${agreement_rules.yolo_dinov2_disagree_cap})`;
      }
    } else if (yolo) {
      source_agreement = 'single_source';
      raw_confidence = yolo.confidence * (source_weights.yolo?.weight ?? 1.0);
      agreement_note = `Fonte única: YOLO ${yolo.confidence.toFixed(3)}`;
    } else if (dinov2) {
      source_agreement = 'single_source';
      raw_confidence = dinov2.confidence * (source_weights.dinov2?.weight ?? 0.95);
      agreement_note = `Fonte única: DINOv2 ${dinov2.confidence.toFixed(3)}`;
    } else {
      source_agreement = 'single_source';
      raw_confidence = 0;
      agreement_note = 'Sem fontes disponíveis';
    }

    if (!thresholds.calibrated) {
      raw_confidence = Math.min(raw_confidence, agreement_rules.uncalibrated_max_auto_accept);
    }

    return { confidence_final: raw_confidence, source_agreement, agreement_note };
  }

  private decide(
    confidence: number,
    thresholds: ClassThreshold,
    border_detection: boolean,
    budgetRemainingCents: number,
  ): { decision: RouterDecision; budget_note: string } {
    let decision = this.mapConfidenceToDecision(confidence, thresholds);
    let budget_note = '';

    if (border_detection) {
      decision = this.downgradeDecision(decision);
    }

    if (decision === 'verify_with_vision') {
      const estimated = this.config.budget.estimated_vision_cost_cents;

      if (budgetRemainingCents < estimated) {
        decision = 'send_to_review';
        budget_note = `Budget insuficiente (restam $${budgetRemainingCents.toFixed(2)}¢, custo estimado $${estimated}¢) → rebaixado para send_to_review`;
      } else if (budgetRemainingCents < this.config.budget.alert_cents) {
        budget_note = `ALERTA: budget em ${((1 - budgetRemainingCents / this.config.budget.hard_cap_cents) * 100).toFixed(0)}% utilizado`;
      }
    }

    return { decision, budget_note };
  }

  private mapConfidenceToDecision(
    confidence: number,
    thresholds: ClassThreshold,
  ): RouterDecision {
    if (confidence >= thresholds.auto_accept) return 'auto_accept';
    if (confidence >= thresholds.verify_with_vision) return 'verify_with_vision';
    if (confidence >= thresholds.reject) return 'send_to_review';
    return 'reject';
  }

  private downgradeDecision(decision: RouterDecision): RouterDecision {
    const ladder: RouterDecision[] = [
      'auto_accept',
      'verify_with_vision',
      'send_to_review',
      'reject',
    ];
    const idx = ladder.indexOf(decision);
    return idx < ladder.length - 1 ? ladder[idx + 1] : decision;
  }

  private buildJustification(
    classSlug: string,
    confidenceFinal: number,
    decision: RouterDecision,
    thresholds: ClassThreshold & { is_default: boolean },
    sourceAgreement: SourceAgreement,
    agreementNote: string,
    budgetNote: string,
  ): string {
    const thresholdSource = thresholds.is_default ? 'defaults' : `config[${classSlug}]`;
    const calibrated = thresholds.calibrated ? 'calibrado' : 'NÃO calibrado';

    let parts = [
      `Classe '${classSlug}': confidence_final=${confidenceFinal.toFixed(3)}`,
      `(${agreementNote}, acordo=${sourceAgreement})`,
      `Threshold ${thresholdSource} [${calibrated}]: auto_accept=${thresholds.auto_accept}, verify=${thresholds.verify_with_vision}, reject=${thresholds.reject}`,
      `Decisão: ${decision}`,
    ];

    if (budgetNote) parts.push(budgetNote);

    return parts.join('. ');
  }
}
