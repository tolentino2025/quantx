# Skill: DINOv2 Few-Shot Matching

## Quando invocar
Quando o tenant tem biblioteca pessoal (`tenant_library_size > 0`) ou quando
YOLO retorna objectness alta mas classification baixa (símbolo fora do modelo
base). Invocada pelo Recognition Orchestrator em paralelo com YOLO.

---

## Por que DINOv2 para SPCI

YOLO conhece apenas as classes do modelo base (~150 classes). Cada engenharia
tem variações de símbolo (ex: sprinkler com marcação da fabricante, extintor
com cor diferente). DINOv2 generaliza por similaridade visual sem precisar
retreinar — basta ter 1 amostra validada da classe nova.

---

## Arquitetura do Embedding

- **Modelo base**: `facebook/dinov2-base` (768 dimensões)
- **Pooling**: CLS token (não average pooling — melhor para símbolos isolados)
- **Normalização**: L2 norm antes de armazenar e antes de comparar
- **Armazenamento**: pgvector com índice HNSW particionado por `tenant_id`
- **Janela de entrada**: 128×128px (padrão), adaptada por escala (ver tabela)

---

## Janela de Entrada por Escala

| Escala | Janela DINOv2 | Resize interno |
|---|---|---|
| 1:50 | 96×96px | → 224×224 |
| 1:100 | 128×128px | → 224×224 |
| 1:200 | 160×160px | → 224×224 |
| 1:250 | 192×192px | → 224×224 |

---

## Fluxo de Match

```python
from transformers import AutoImageProcessor, AutoModel
import torch, torch.nn.functional as F

processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
model = AutoModel.from_pretrained("facebook/dinov2-base")

def embed(crop_image):
    inputs = processor(images=crop_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]  # CLS token
    return F.normalize(embedding, dim=-1).squeeze().numpy()

def query_top_k(query_embedding, tenant_id, k=5):
    # Consulta pgvector — retorna top-k classes mais similares
    # Filtrado por tenant_id para isolamento
    pass
```

---

## Thresholds de Similaridade

| Cosine Similarity | Decisão |
|---|---|
| ≥ 0.90 | Match forte — passar para ConfidenceRouter como `dinov2: high` |
| 0.75 – 0.89 | Match moderado — passar como `dinov2: medium` |
| 0.60 – 0.74 | Match fraco — passar como `dinov2: low` (provavelmente verify) |
| < 0.60 | Sem match — classe desconhecida para este tenant |

Thresholds calibrados por classe em `config/confidence-thresholds.json`
têm precedência sobre os defaults acima.

---

## Re-embedding Após Correção

Quando Dataset Curation Agent adiciona nova amostra:

```python
# 1. Gerar embedding do crop
new_embedding = embed(crop_image)

# 2. Upsert no pgvector
# INSERT INTO embeddings (tenant_id, class_slug, embedding, sample_id)
# VALUES ($1, $2, $3::vector, $4)
# ON CONFLICT (tenant_id, sample_id) DO UPDATE SET embedding = $3::vector

# 3. Invalidar cache de índice HNSW para o tenant
# (reconstrução automática no próximo query ou forçar rebuild)
```

Latência esperada de re-embedding: < 3s por amostra em GPU.

---

## Acordo com YOLO

Quando YOLO e DINOv2 retornam para a mesma detecção:

| Situação | Resultado para ConfidenceRouter |
|---|---|
| Mesma classe, ambos acima de threshold | `agreement: agree`, boost +0.10 |
| Classes diferentes (qualquer confiança) | `agreement: disagree`, cap em 0.65 |
| Só DINOv2 (classe fora do YOLO) | `agreement: single_source` |
| DINOv2 < 0.60 (sem match) | Não reportar — ignorar silenciosamente |

---

## Isolamento Multi-Tenant

- Query SEMPRE filtra por `tenant_id` — nunca vazar embeddings entre tenants
- Índice pgvector: `CREATE INDEX ON embeddings USING hnsw (embedding vector_cosine_ops) WHERE tenant_id = $1`
- Tenant com biblioteca vazia: DINOv2 não roda (Orchestrator verifica `tenant_library_size`)

---

_Skill: dinov2-fewshot | Camada: L2 | Versão: 1.0 | 2026-05-09_
