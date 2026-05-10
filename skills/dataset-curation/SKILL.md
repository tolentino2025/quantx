# Skill: Dataset Curation

## Quando invocar
Após correção humana validada (com reviewer_id e timestamp). Invocada pelo
Dataset Curation Agent. Também consultada pelo False Positive Review Agent
para estimar cobertura de classes.

---

## Geração de Crop

```python
from PIL import Image

def generate_crop(tile_image_path, bbox, padding_ratio=0.20, min_size=64):
    img = Image.open(tile_image_path)
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    # Padding proporcional
    px, py = int(w * padding_ratio), int(h * padding_ratio)
    crop = img.crop((
        max(0, x1 - px), max(0, y1 - py),
        min(img.width, x2 + px), min(img.height, y2 + py)
    ))

    # Garantir tamanho mínimo para DINOv2
    if min(crop.size) < min_size:
        scale = min_size / min(crop.size)
        crop = crop.resize(
            (int(crop.width * scale), int(crop.height * scale)),
            Image.LANCZOS
        )
    return crop
```

---

## Augmentations por Invariância

```python
from PIL import Image, ImageEnhance
import random

def augment(crop, invariances):
    augmented = [crop]  # sempre incluir original

    if invariances.get("rotation"):
        for angle in [90, 180, 270]:
            augmented.append(crop.rotate(angle, expand=True))

    if invariances.get("flip"):
        augmented.append(crop.transpose(Image.FLIP_LEFT_RIGHT))

    if invariances.get("scale"):
        for factor in [0.8, 1.2]:
            new_size = (int(crop.width * factor), int(crop.height * factor))
            augmented.append(crop.resize(new_size, Image.LANCZOS))

    # brightness: sempre aplicar (invariante em todas as classes)
    for factor in [0.8, 1.2]:
        enhancer = ImageEnhance.Brightness(crop)
        augmented.append(enhancer.enhance(factor))

    return augmented
```

---

## Formato de Label YOLO

```python
def to_yolo_label(bbox_page, page_size, class_id):
    # bbox: [x1, y1, x2, y2] em pixels absolutas
    # page_size: (width, height)
    w, h = page_size
    x1, y1, x2, y2 = bbox_page

    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h

    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
```

Salvar em `dataset/{tenant_id}/labels/{sample_id}.txt`.

---

## Estrutura de Armazenamento

```
dataset/
├── {tenant_id}/
│   ├── images/
│   │   ├── {sample_id}.png          ← crop original
│   │   └── {sample_id}_aug_r90.png  ← augmented
│   ├── labels/
│   │   └── {sample_id}.txt          ← YOLO label
│   ├── metadata/
│   │   └── {sample_id}.json         ← trail completa
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       └── test.txt                  ← 70/20/10
```

---

## Mapeamento Ação → Tipo de Amostra

| Action | Tipo | Classe |
|---|---|---|
| `accept` | positivo | `original_class` |
| `reject` | negativo (`background`) | nenhuma |
| `reclassify` | positivo | `corrected_class` |
| `reclassify` | hard_negative | `original_class` |
| `adjust_bbox` | positivo | `original_class` (com `corrected_bbox`) |
| `add_new` | positivo | nova classe (após Catalog Review) |

---

## Threshold de Fine-Tune

| Condição | Ação |
|---|---|
| `new_samples >= 200` | Agendar fine-tune imediato |
| Sexta-feira 22h AND `new_samples >= 50` | Agendar fine-tune semanal |
| `new_samples < 50` | Apenas re-embed DINOv2, sem fine-tune |

---

## Separação de Splits

```python
import random

def split_samples(samples, train=0.70, val=0.20, test=0.10, seed=42):
    random.seed(seed)
    random.shuffle(samples)
    n = len(samples)
    t = int(n * train)
    v = int(n * (train + val))
    return samples[:t], samples[t:v], samples[v:]
```

NUNCA misturar splits entre tenants. NUNCA usar amostras de tenant
individual no dataset do modelo base sem consentimento explícito.

---

## Metadados Obrigatórios por Amostra

```json
{
  "sample_id": "uuid",
  "correction_id": "uuid",
  "detection_id": "uuid",
  "plan_id": "uuid",
  "tenant_id": "uuid",
  "class_slug": "sprinkler-pendent-k57",
  "type": "positive | negative | hard_negative",
  "action": "accept | reject | reclassify | adjust_bbox | add_new",
  "reviewer_id": "hash-anonimizado",
  "source_model_version": "yolov15-spci-2026.04",
  "augmentations_applied": ["original", "brightness_0.8", "brightness_1.2"],
  "created_at": "2026-05-09T12:35:10Z"
}
```

---

## Relatório de Cobertura

```python
def coverage_status(sample_count):
    if sample_count < 5:   return "critical"
    if sample_count < 20:  return "insufficient"
    if sample_count < 50:  return "adequate"
    return "strong"
```

Classes com status `critical` ou `insufficient` devem aparecer no
relatório semanal para o engenheiro cadastrar mais amostras.

---

_Skill: dataset-curation | Camada: L2 | Versão: 1.0 | 2026-05-09_
