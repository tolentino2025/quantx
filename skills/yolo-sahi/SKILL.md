# Skill: YOLO + SAHI Inference

## Quando invocar
Ao executar inferência YOLO em planta SPCI completa. Fornece configuração
de tiles, parâmetros de NMS e batching. Invocada pelo Recognition Orchestrator
para montar o plano de inferência.

---

## Por que SAHI para SPCI

Plantas A1 em 300dpi têm ~7000×10000px. YOLO padrão redimensiona a entrada
para 640×1280px — símbolos de 35px viram 2px e somem. SAHI fatia a imagem
em tiles com overlap, roda YOLO em cada tile no tamanho nativo e faz merge.

---

## Configuração de Tiles por Escala

| Escala | DPI | Tile (px) | Overlap | imgsz | Batch size |
|---|---|---|---|---|---|
| 1:50 | 200 | 640×640 | 15% | 640 | 16 |
| 1:75 | 250 | 768×768 | 18% | 768 | 12 |
| 1:100 | 300 | 1024×1024 | 20% | 1024 | 8 |
| 1:150 | 350 | 1024×1024 | 22% | 1024 | 8 |
| 1:200 | 400 | 1280×1280 | 25% | 1280 | 4 |
| 1:250 | 450 | 1280×1280 | 25% | 1280 | 4 |

---

## Parâmetros de Inferência Recomendados

```python
results = model.predict(
    source=tile_image,
    imgsz=1024,
    conf=0.25,        # baixo — ConfidenceRouter filtra depois
    iou=0.45,         # NMS dentro do tile
    augment=False,    # não usar TTA em produção (custo)
    verbose=False
)
```

> `conf=0.25` é intencional — o ConfidenceRouter aplica thresholds por classe.
> Não elevar aqui ou detecções legítimas de baixa confiança serão perdidas.

---

## Uso com SAHI

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="models/yolov15-spci-2026.04.pt",
    confidence_threshold=0.25,
    device="cuda:0"
)

result = get_sliced_prediction(
    image=page_image_path,
    detection_model=model,
    slice_height=1024,
    slice_width=1024,
    overlap_height_ratio=0.20,
    overlap_width_ratio=0.20,
    perform_standard_pred=False,   # já cobrimos com tiles
    postprocess_type="NMM",        # Non-Maximum Merging (melhor que NMS para símbolos)
    postprocess_match_threshold=0.45
)
```

---

## Mascaramento da Região de Legenda

Antes de rodar SAHI, mascarar a região de legenda identificada pelo
Scale Detection Agent. Preencher com branco para evitar detecções falsas.

```python
from PIL import Image, ImageDraw

img = Image.open(page_image_path)
if legend_region:
    draw = ImageDraw.Draw(img)
    draw.rectangle(legend_region, fill=(255, 255, 255))
```

---

## Saída por Tile

```json
{
  "tile_id": "uuid",
  "tile_origin": [0, 0],
  "tile_size": [1024, 1024],
  "detections": [
    {
      "class_id": 14,
      "class_slug": "sprinkler-pendent-k57",
      "bbox_tile": [412, 387, 447, 422],
      "confidence": 0.81,
      "source": "yolo",
      "model_version": "yolov15-spci-2026.04"
    }
  ]
}
```

---

## Classes Conhecidas — Mapeamento de IDs

IDs de classe são definidos em `models/classes.json` do modelo ativo.
NUNCA hardcodar IDs — sempre carregar do arquivo de classes do modelo.
Validar na inicialização que o número de classes no `.pt` bate com o arquivo.

---

## Erros Comuns

| Problema | Causa | Solução |
|---|---|---|
| Detecções apenas no centro da planta | Tile sem overlap suficiente | Aumentar overlap para 25% |
| Muitas duplicatas | NMS muito permissivo | Baixar iou para 0.35 |
| Símbolo não detectado em alta escala (1:50) | imgsz muito grande para GPU | Reduzir tile para 640, aumentar batch |
| OOM na GPU | Batch size alto com imgsz 1280 | Reduzir batch para 2 |

---

_Skill: yolo-sahi | Camada: L2 | Versão: 1.0 | 2026-05-09_
