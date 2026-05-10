# Skill: Symbol Catalog

## Quando invocar
Ao adicionar, editar, buscar ou validar classes na biblioteca (base ou pessoal).
Invocada pelo Catalog Review Agent e pelo Dataset Curation Agent.

---

## Schema Canônico de Classe

```json
{
  "slug": "sprinkler-pendent-k57",
  "className": "SprinklerPendentK57",
  "superclass": "sprinkler/pendent",
  "visual_description": "Círculo com cruz interna, ramo vertical descendente, fator K=5.7. Geralmente desenhado em escala 1:50 com ~3mm.",
  "invariances": {
    "rotation": false,
    "flip": false,
    "scale": true
  },
  "reference_image_url": "s3://quantx-catalog/{tenant_id}/{slug}.png",
  "nbr_reference": "NBR 10897:2020",
  "tenant_id": null,
  "created_at": "2026-05-09T00:00:00Z",
  "updated_at": "2026-05-09T00:00:00Z",
  "deleted_at": null,
  "version": 1
}
```

---

## Regras de Slug

| Regra | Válido | Inválido |
|---|---|---|
| kebab-case | `extintor-co2-5kg` | `ExtintorCO2` |
| Sem nome comercial | `sprinkler-pendent-k57` | `viking-pendant-vk302` |
| Sem caracteres especiais | `hidrante-parede-tipo2` | `hidrante_parede/tipo2` |
| 5–60 chars | ok | slug de 3 chars |
| Único por tenant | verificar antes de criar | — |

---

## Geração Automática de Slug

```python
import re, unicodedata

def generate_slug(superclass, characteristics):
    # superclass: "sprinkler/pendent"
    # characteristics: ["k57", "concealed"]
    base = superclass.replace("/", "-")
    suffix = "-".join(
        unicodedata.normalize("NFKD", c)
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .strip()
        for c in characteristics
    )
    slug = f"{base}-{suffix}" if suffix else base
    slug = re.sub(r"[^a-z0-9-]", "-", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:60]
```

---

## Superclasses Disponíveis

Carregar sempre de `config/superclasses.json`. NUNCA usar lista hardcoded.

Superclasses atuais:
- `sprinkler/{pendent,upright,sidewall,concealed,dry}`
- `extintor/{pqs,co2,agua,affs,classe-k}`
- `hidrante/{parede,recalque,coluna-seca}`
- `detector/{fumaca,temperatura,multicriterio,gas,beam}`
- `alarme/{manual,sirene,central,modulo}`
- `valvula/{governadora,retencao,gaveta,recalque}`
- `bomba/{principal,jockey,diesel}`
- `mangotinho/{parede}`
- `iluminacao-emergencia/{bloco,pictograma-saida}`

---

## Invariâncias por Superclasse — Defaults

| Superclasse | rotation | flip | scale |
|---|---|---|---|
| sprinkler/* | false | false | true |
| extintor/* | true | true | true |
| hidrante/parede | false | false | true |
| hidrante/recalque | true | true | true |
| detector/* | true | true | true |
| alarme/manual | false | false | true |
| alarme/sirene | true | true | true |
| valvula/* | false | false | true |
| bomba/* | false | false | true |

> Defaults podem ser sobrescritos por classe individual.

---

## Soft Delete

```python
# NUNCA deletar fisicamente uma classe
# Sempre usar soft delete:
{
  "deleted_at": "2026-05-09T00:00:00Z",
  "deleted_by": "reviewer-hash-anonimizado",
  "deletion_reason": "duplicata de sprinkler-pendent-k57"
}
```

Classe com `deleted_at != null`:
- Não aparece em buscas de inferência
- Mantida no histórico de detecções antigas
- Não pode ser recriada com o mesmo slug (verificar incluindo deletadas)

---

## Busca por Similaridade

Para detectar duplicatas antes de criar classe nova:

```sql
SELECT class_slug, 1 - (embedding <=> $1::vector) AS similarity
FROM catalog_embeddings
WHERE tenant_id = $2
  AND deleted_at IS NULL
ORDER BY embedding <=> $1::vector
LIMIT 10;
```

Threshold de alerta: similarity ≥ 0.85.

---

## Cobertura Mínima por Classe

Para o DINOv2 funcionar bem em few-shot:

| Cobertura | Status | Ação |
|---|---|---|
| < 5 amostras | Crítico | DINOv2 não habilita para esta classe |
| 5–19 amostras | Insuficiente | DINOv2 habilita com threshold mais restritivo (0.85) |
| 20–49 amostras | Adequado | DINOv2 padrão |
| ≥ 50 amostras | Forte | Threshold pode ser relaxado (0.72) |

---

_Skill: symbol-catalog | Camada: L2 | Versão: 1.0 | 2026-05-09_
