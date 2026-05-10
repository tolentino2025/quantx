# Skill: Audit Reports

## Quando invocar
Ao gerar trilha de auditoria de uma detecção, relatório de planta processada,
ou histórico de decisões do ConfidenceRouter. Toda detecção persistida deve
ter trail gerado por esta skill.

---

## Trail Completo de Detecção

```json
{
  "detection_id": "uuid",
  "plan_id": "uuid",
  "tenant_id": "uuid",
  "page_number": 1,
  "tile_id": "uuid",
  "class_slug": "sprinkler-pendent-k57",
  "bbox": [412, 387, 447, 422],
  "bbox_page": [1436, 3971, 1471, 4006],

  "pipeline": {
    "scale_detected": { "scale": "1:100", "method": "ocr_cartouche", "confidence": 0.92 },
    "tile_config": { "tile_size": 1024, "overlap": 0.20, "dpi": 300 },
    "yolo": { "confidence": 0.81, "model_version": "yolov15-spci-2026.04" },
    "dinov2": { "confidence": 0.88, "top1_slug": "sprinkler-pendent-k57", "similarity": 0.88 },
    "agreement": "agree",
    "nms": { "action": "kept", "border_detection": false }
  },

  "router": {
    "confidence_final": 0.91,
    "threshold_used": { "auto_accept": 0.87, "source": "config_per_class", "calibrated": true },
    "decision": "auto_accept",
    "justification": "YOLO 0.81 + DINOv2 0.88 concordam (+0.10 boost). confidence_final=0.91 >= auto_accept=0.87.",
    "cost_cents": 0.0
  },

  "vision": null,

  "human_review": null,

  "timestamp": "2026-05-09T12:34:56Z",
  "model_version": "yolov15-spci-2026.04"
}
```

---

## Trail com Vision Verification

```json
{
  "vision": {
    "model": "claude-sonnet-4-6",
    "cost_cents": 2.1,
    "latency_ms": 2340,
    "prompt_version": "v2.1",
    "result": {
      "is_correct_class": true,
      "confidence": 0.94,
      "bbox_quality": "good",
      "reasoning": "Símbolo apresenta círculo com cruz interna e ramo descendente característico do pendent."
    }
  }
}
```

---

## Relatório de Planta Processada

```json
{
  "plan_id": "uuid",
  "tenant_id": "uuid",
  "processed_at": "2026-05-09T12:35:00Z",
  "pages": 3,
  "processing_time_s": 28.4,

  "summary": {
    "total_detections": 142,
    "auto_accepted": 118,
    "verified_with_vision": 14,
    "sent_to_review": 8,
    "rejected": 2,
    "vision_cost_cents": 28.7,
    "budget_used_pct": 57.4
  },

  "by_class": {
    "sprinkler-pendent-k57": { "count": 87, "avg_confidence": 0.89 },
    "extintor-pqs": { "count": 12, "avg_confidence": 0.84 },
    "hidrante-parede": { "count": 6, "avg_confidence": 0.91 },
    "detector-fumaca": { "count": 37, "avg_confidence": 0.86 }
  },

  "warnings": [
    "budget_alert: 57.4% do cap utilizado",
    "2 classes sem threshold calibrado: [detector-gas, alarme-modulo]"
  ],

  "scale_info": { "scale": "1:100", "confidence": 0.92, "method": "ocr_cartouche" }
}
```

---

## Relatório Semanal de Qualidade

Gerado todo domingo pelo Dataset Curation Agent:

```json
{
  "period": "2026-05-03 to 2026-05-09",
  "tenant_id": "uuid",

  "volume": {
    "plans_processed": 23,
    "detections_total": 3847,
    "corrections_received": 198
  },

  "accuracy": {
    "auto_accept_rate": 0.812,
    "false_positive_rate": 0.062,
    "vision_usage_rate": 0.098,
    "avg_cost_per_plan_cents": 31.2
  },

  "classes_needing_attention": [
    {
      "class_slug": "extintor-pqs",
      "issue": "fp_rate_high",
      "fp_rate": 0.091,
      "recommendation": "Elevar threshold auto_accept para 0.90"
    }
  ],

  "dataset_growth": {
    "new_samples": 198,
    "by_class": { "sprinkler-pendent-k57": 89, "extintor-pqs": 43 },
    "finetune_triggered": false,
    "finetune_threshold": 200,
    "current_count": 198
  }
}
```

---

## Campos Obrigatórios em Toda Trail

NUNCA persistir detecção sem:
- `source` (yolo | dinov2 | rtdetr | vision | template)
- `model_version`
- `confidence` (valor final após ajustes do router)
- `router_decision`
- `router_justification`
- `timestamp`

---

_Skill: audit-reports | Camada: L2 | Versão: 1.0 | 2026-05-09_
