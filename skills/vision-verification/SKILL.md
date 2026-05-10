# Skill: Vision Verification

## Quando invocar
Apenas quando ConfidenceRouter decidir `verify_with_vision`. NUNCA chamar
Vision API diretamente sem passar pelo router. Budget hard cap: $0.50/planta.

---

## Modelos Disponíveis (em ordem de preferência)

| Modelo | Custo/imagem | Velocidade | Usar quando |
|---|---|---|---|
| Claude claude-sonnet-4-6 | ~$0.02 | 2–4s | Padrão — melhor reasoning sobre símbolos |
| GPT-4o | ~$0.03 | 2–5s | Fallback se Claude indisponível |
| Gemini 1.5 Flash | ~$0.01 | 1–3s | Fallback barato para lotes grandes |

---

## Prompt Padrão — Verificação de Símbolo SPCI

```
Você é um especialista em sistemas de prevenção e combate a incêndio (SPCI)
brasileiro. Analise o recorte de planta técnica abaixo e responda:

1. O símbolo no centro da imagem é um(a) {class_name} ({visual_description})?
2. Se não for, qual símbolo SPCI mais se parece?
3. O bbox delimitado está correto (símbolo completamente dentro)?

Responda APENAS no formato JSON:
{
  "is_correct_class": true | false,
  "confidence": 0.0–1.0,
  "suggested_class": "{class_slug} ou null se correto",
  "bbox_quality": "good | tight | loose | wrong",
  "reasoning": "explicação em 1 frase"
}

Contexto: planta em escala {scale}, DPI {dpi}. Símbolo esperado tem ~{symbol_size_px}px.
```

---

## Prompt para Classe Desconhecida

```
Você é um especialista em SPCI brasileiro. Este recorte de planta técnica
contém um símbolo não identificado pelo modelo automático.

Identifique:
1. A qual categoria pertence: {superclasses_list}
2. Qual NBR rege este equipamento
3. Descreva o símbolo visualmente (forma, elementos internos, orientação)

Formato de resposta JSON:
{
  "superclass": "{superclass_slug}",
  "suggested_slug": "nome-kebab-case-descritivo",
  "visual_description": "descrição em PT-BR, 50-200 chars",
  "nbr_reference": "NBR XXXXX:AAAA ou null",
  "confidence": 0.0–1.0,
  "reasoning": "explicação em 1–2 frases"
}
```

---

## Preparação do Crop para Vision

```python
def prepare_crop_for_vision(tile_image, bbox, padding_ratio=0.3):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    # Padding proporcional ao tamanho do símbolo
    px, py = int(w * padding_ratio), int(h * padding_ratio)

    crop = tile_image.crop((
        max(0, x1 - px),
        max(0, y1 - py),
        min(tile_image.width, x2 + px),
        min(tile_image.height, y2 + py)
    ))

    # Resize para mínimo 256px no lado menor (Vision precisa de detalhe)
    min_side = min(crop.size)
    if min_side < 256:
        scale = 256 / min_side
        crop = crop.resize(
            (int(crop.width * scale), int(crop.height * scale)),
            Image.LANCZOS
        )

    return crop
```

---

## Controle de Custo por Lote

Antes de cada chamada Vision, verificar budget restante:

```python
HARD_CAP_CENTS = 50.0
ALERT_CENTS = 30.0

def can_call_vision(budget_used_cents, estimated_cost_cents):
    if budget_used_cents + estimated_cost_cents > HARD_CAP_CENTS:
        return False, "budget_exceeded"
    if budget_used_cents > ALERT_CENTS:
        return True, "budget_alert"  # permitir mas avisar
    return True, "ok"
```

Estratégia quando budget está se esgotando:
1. Priorizar detecções com `confidence` entre 0.70–0.84 (mais impacto)
2. Detecções com `confidence < 0.65` → `send_to_review` direto (sem Vision)
3. Detecções com `confidence >= 0.85` → auto_accept sem Vision

---

## Parsing da Resposta Vision

```python
import json, re

def parse_vision_response(response_text):
    # Extrair JSON mesmo se vier com texto ao redor
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if not match:
        return {"error": "no_json_in_response", "raw": response_text[:200]}
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return {"error": "invalid_json", "raw": match.group()[:200]}
```

---

## Registro de Custo na Trail

Toda chamada Vision DEVE registrar custo na detecção:

```json
{
  "detection_id": "uuid",
  "vision_call": {
    "model": "claude-sonnet-4-6",
    "cost_cents": 2.1,
    "latency_ms": 2340,
    "result": "is_correct_class: true, confidence: 0.94"
  }
}
```

---

_Skill: vision-verification | Camada: L2 | Versão: 1.0 | 2026-05-09_
