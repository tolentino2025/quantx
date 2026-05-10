# Skill: PDF Rendering

## Quando invocar
Antes de qualquer inferência em planta PDF nova. Define DPI, modo de cor
e separação de páginas. Sempre invocada pelo Scale Detection Agent antes
de recomendar parâmetros de tile.

---

## Parâmetros por Escala

| Escala | DPI render | Modo cor | Formato saída | Tamanho A1 aprox. |
|---|---|---|---|---|
| 1:50 | 200 | RGB | PNG | 4960×7016px |
| 1:75 | 250 | RGB | PNG | 6200×8770px |
| 1:100 | 300 | RGB | PNG | 7433×10512px |
| 1:150 | 350 | RGB | PNG | 8672×12264px |
| 1:200 | 400 | RGB | PNG | 9921×14031px |
| 1:250 | 450 | RGB | PNG | 11160×15785px |
| desconhecida | 300 | RGB | PNG | — |

> Usar sempre RGB (não grayscale) — linhas de planta CAD usam cores para
> distinguir camadas; grayscale perde essa informação.

---

## Pré-processamento Obrigatório

1. **Detecção de orientação**: verificar se a página está em retrato ou
   paisagem. Plantas A1/A0 são quase sempre paisagem. Rotacionar se necessário.

2. **Remoção de borda branca**: crop automático das margens em branco
   (`PIL.ImageOps.crop` com threshold=250) para não desperdiçar tiles.

3. **Normalização de contraste**: se histograma indicar planta com fundo
   muito escuro ou muito claro (ex: plotagens com fundo cinza), aplicar
   `ImageEnhance.Contrast` fator 1.2.

4. **Separação de páginas**: plantas multi-página devem ter cada página
   processada individualmente. Não concatenar páginas.

---

## Região do Cartouche — Não processar como planta

O cartouche (canto inferior direito, ~15% da área) contém texto, tabelas
e símbolos de referência que NÃO são equipamentos SPCI. Identificar e
excluir da inferência principal (passar para Scale Detection Agent).

```python
# Estimativa de região do cartouche
cartouche_region = (
    int(width * 0.72),   # x1
    int(height * 0.82),  # y1
    width,               # x2
    height               # y2
)
```

---

## Biblioteca Recomendada

```python
from pdf2image import convert_from_path
from PIL import Image, ImageOps, ImageEnhance

pages = convert_from_path(
    pdf_path,
    dpi=300,
    fmt="png",
    thread_count=4,
    use_cropbox=True
)
```

---

## Saída Esperada

```json
{
  "plan_id": "uuid",
  "pages": [
    {
      "page_number": 1,
      "image_path": "tmp/{plan_id}/page_001_300dpi.png",
      "width_px": 7433,
      "height_px": 10512,
      "dpi": 300,
      "orientation": "landscape",
      "cartouche_region": [5352, 8620, 7433, 10512]
    }
  ]
}
```

---

## Erros Comuns em Plantas SPCI

| Problema | Sintoma | Correção |
|---|---|---|
| PDF vetorial sem rasterização | Imagem em branco | Usar `--use-cropbox` + `dpi mínimo 150` |
| Planta em preto e branco apenas | Perda de camadas | Manter RGB mesmo assim |
| PDF com múltiplas plantas por página | Símbolos duplicados | Detecção manual de plantas internas (fora do escopo desta skill) |
| Planta escaneada (raster) | Baixa qualidade | Aplicar deskew antes da inferência |

---

_Skill: pdf-rendering | Camada: L2 | Versão: 1.0 | 2026-05-09_
