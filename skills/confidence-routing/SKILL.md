# Skill: Confidence Routing

## Quando invocar
Ao tomar qualquer decisão de destino de detecção (auto_accept, verify_with_vision,
send_to_review, reject). Invocada pelo Confidence Router Agent. Também consultada
pelo False Positive Review Agent para estimar impacto de mudança de threshold.

---

## Arquivo de Configuração

`config/confidence-thresholds.json` — thresholds por classe e por fonte.

```json
{
  "defaults": {
    "auto_accept": 0.85,
    "verify_with_vision": 0.65,
    "reject": 0.40
  },
  "by_class": {
    "sprinkler-pendent-k57": {
      "auto_accept": 0.87,
      "verify_with_vision": 0.68,
      "reject": 0.40,
      "calibrated": true,
      "last_calibration": "2026-05-01",
      "sample_size": 1240
    },
    "extintor-pqs": {
      "auto_accept": 0.90,
      "verify_with_vision": 0.70,
      "reject": 0.45,
      "calibrated": true,
      "last_calibration": "2026-04-28",
      "sample_size": 847
    }
  },
  "by_source": {
    "yolo": { "weight": 1.0 },
    "dinov2": { "weight": 0.95 },
    "rtdetr": { "weight": 1.05 },
    "template": { "weight": 0.70 }
  }
}
```

---

## Algoritmo de Decisão

```
1. Carregar thresholds da classe (ou defaults se não calibrada)
2. Aplicar peso da fonte: confidence_weighted = confidence * source_weight
3. Aplicar ajuste de acordo:
   - YOLO + DINOv2 concordam → +0.10 (cap 0.99)
   - YOLO + DINOv2 discordam → cap em 0.65
4. Aplicar penalidade de border_detection:
   - border_detection=true → rebaixar decisão um nível
5. Mapear confidence_final para decisão:
   - >= auto_accept  → auto_accept
   - >= verify       → verify_with_vision
   - >= reject       → send_to_review
   - < reject        → reject
6. Verificar budget antes de confirmar verify_with_vision
```

---

## Tabela de Penalidades e Bônus

| Condição | Ajuste | Justificativa |
|---|---|---|
| YOLO + DINOv2 concordam | +0.10 | Dois modelos independentes concordam |
| YOLO + DINOv2 discordam | cap 0.65 | Ambiguidade — humano ou Vision devem decidir |
| `border_detection: true` | −1 nível | Bbox incompleto compromete classificação |
| Classe `calibrated: false` | max auto_accept = 0.90 | Threshold não validado com dados reais |
| Fonte `template` | ×0.70 | Template Matching só em audit_mode |

---

## Controle de Budget na Decisão

```python
def route_with_budget(confidence_final, thresholds, budget_remaining_cents):
    base_decision = map_to_decision(confidence_final, thresholds)

    if base_decision == "verify_with_vision":
        estimated_cost = 2.5  # cents, estimativa conservadora

        if budget_remaining_cents < estimated_cost:
            return "send_to_review", "budget_cap_reached"

        if budget_remaining_cents < 30.0:
            # Budget em alerta — só verifica se confiança está na faixa crítica
            if confidence_final < thresholds["verify_with_vision"] + 0.10:
                return "send_to_review", "budget_alert_downgrade"

    return base_decision, "ok"
```

---

## Justificativa Obrigatória

Toda decisão deve gerar string de justificativa para o trail:

```python
def build_justification(class_slug, confidence_final, decision, source, agreement, budget_note):
    return (
        f"Classe '{class_slug}': confidence_final={confidence_final:.3f} "
        f"(fonte={source}, acordo={agreement}). "
        f"Threshold auto_accept={thresholds['auto_accept']:.2f}, "
        f"verify={thresholds['verify_with_vision']:.2f}. "
        f"Decisão: {decision}. {budget_note}"
    )
```

---

## Calibração de Threshold

Threshold de uma classe é considerado **calibrado** quando:
- Dataset de validação ≥ 500 amostras com rótulo humano
- Calculado via precision-recall curve com F1 máximo no operating point
- Revisado após cada fine-tune do YOLO

Classe `calibrated: false` → warn no output do router.

---

_Skill: confidence-routing | Camada: L2 | Versão: 1.0 | 2026-05-09_
