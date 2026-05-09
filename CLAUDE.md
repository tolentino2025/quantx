# QuantX — Regras do Projeto

> Este arquivo é a camada de memória do Claude Code para o QuantX. Toda
> sessão começa lendo este arquivo. Mudanças aqui afetam o comportamento
> de TODOS os agentes, skills e hooks. Edite com critério.

---

## 1. Contexto do Produto

QuantX é uma plataforma de takeoff (quantificação) de projetos de SPCI
(Sistemas de Prevenção e Combate a Incêndio) para engenheiros brasileiros.

O usuário sobe uma planta em PDF, o pipeline reconhece os símbolos de
equipamentos (sprinklers, hidrantes, extintores, detectores, etc.),
quantifica e gera orçamento. O engenheiro valida com 1 clique e as
correções alimentam o aprendizado contínuo do workspace.

Diferenciais do produto:
- Multi-tenant: cada engenharia tem sua biblioteca pessoal de símbolos
- Rastreabilidade: toda detecção tem trilha auditável
- Aprendizado contínuo: correção humana vira treino
- Conformidade com NBRs e ITs estaduais brasileiras

---

## 2. Domínio SPCI — Normas Aplicáveis

| Norma | Aplicação |
|---|---|
| NBR 10897 | Sistemas de chuveiros automáticos (sprinklers) |
| NBR 13714 | Hidrantes e mangotinhos |
| NBR 17240 | Detecção e alarme de incêndio |
| NBR 12693 | Extintores portáteis |
| NBR 15808/15809 | Extintores classe ABC, BC, K |
| NBR 16981 | Sistemas de espuma (AFFS) |
| IT-CBM-SP/RJ/MG | Instruções Técnicas estaduais (variam por UF) |

Quando uma classe nova for adicionada, registrar a NBR de referência no
campo `nbr_reference` do schema canônico (ver §5).

---

## 3. Glossário Mínimo

- **SPCI**: Sistema de Prevenção e Combate a Incêndio
- **Takeoff**: quantificação de itens em projeto para orçamento
- **Cartouche / selo**: bloco no canto da prancha com escala, autor, data
- **Legenda**: lista de símbolos da planta (NÃO confundir com símbolo da planta)
- **Tile**: recorte de imagem para inferência YOLO
- **Trail**: trilha de auditoria de uma detecção
- **Tenant**: cliente/workspace (cada engenharia)
- **Biblioteca pessoal**: símbolos cadastrados pelo tenant (não estão no YOLO base)

---

## 4. Taxonomia — Superclasses Obrigatórias

Toda classe DEVE pertencer a uma superclasse. Lista canônica (versionada
em `config/superclasses.json`):

```
sprinkler/
  ├── pendent          (descendente)
  ├── upright          (ascendente)
  ├── sidewall         (lateral)
  ├── concealed        (oculto)
  └── dry              (seco / antifreeze)

extintor/
  ├── pqs              (pó químico seco)
  ├── co2              (gás carbônico)
  ├── agua             (água pressurizada)
  ├── affs             (espuma)
  └── classe-k         (cozinha)

hidrante/
  ├── parede           (tipo armário)
  ├── recalque         (passeio público)
  └── coluna-seca      (sem água permanente)

detector/
  ├── fumaca           (óptico/iônico)
  ├── temperatura      (termovelocimétrico)
  ├── multicriterio    (fumaça + temperatura)
  ├── gas              (CO, GLP, GN)
  └── beam             (linear / feixe)

alarme/
  ├── manual           (acionador manual)
  ├── sirene           (audiovisual)
  ├── central          (CPDA)
  └── modulo           (entrada/saída endereçável)

valvula/
  ├── governadora      (alarme de fluxo)
  ├── retencao
  ├── gaveta
  └── recalque         (registro de recalque)

bomba/
  ├── principal
  ├── jockey
  └── diesel

mangotinho/
  └── parede

iluminacao-emergencia/
  ├── bloco
  └── pictograma-saida
```

NUNCA criar superclasse nova sem atualizar `config/superclasses.json` E
revisar com engenheiro SPCI.

---

## 5. Schema Canônico de Classe

Toda classe na biblioteca (base ou pessoal) DEVE seguir este schema:

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
  "reference_image_url": "s3://quantx-catalog/{tenant}/{slug}.png",
  "nbr_reference": "NBR 10897:2020",
  "tenant_id": null,
  "created_at": "2026-05-09T00:00:00Z",
  "version": 1
}
```

Regras:
- `slug`: kebab-case, único por tenant (ou global se `tenant_id=null`)
- `className`: PascalCase, ≤ 40 chars
- `visual_description`: PT-BR, 50–300 chars, descritivo (não comercial)
- `invariances`: usado pelo Dataset Curation para gerar augmentations válidas
- `tenant_id=null` significa classe do modelo base (compartilhada)

---

## 6. Schema Canônico de Detecção

Toda detecção emitida pelo pipeline DEVE carregar trilha completa:

```json
{
  "detection_id": "uuid",
  "plan_id": "uuid",
  "tile_id": "uuid",
  "class_slug": "sprinkler-pendent-k57",
  "bbox": [x1, y1, x2, y2],
  "confidence": 0.87,
  "source": "yolo|dinov2|rtdetr|vision|template",
  "model_version": "yolov15-spci-2026.04",
  "agreement": {
    "yolo": 0.87,
    "dinov2": 0.91
  },
  "border_detection": false,
  "router_decision": "auto_accept|verify_with_vision|send_to_review|reject",
  "router_justification": "...",
  "cost_cents": 0.02,
  "timestamp": "2026-05-09T12:34:56Z"
}
```

NUNCA persistir detecção sem `source`, `model_version`, `confidence` e
`router_decision`.

---

## 7. Schema Canônico de Correção Humana

```json
{
  "correction_id": "uuid",
  "detection_id": "uuid",
  "reviewer_id": "anonymized-hash",
  "action": "accept|reject|reclassify|adjust_bbox|add_new",
  "original_class": "sprinkler-pendent-k57",
  "corrected_class": "sprinkler-upright-k57",
  "corrected_bbox": [x1, y1, x2, y2],
  "reason": "símbolo era ascendente, não descendente",
  "timestamp": "2026-05-09T12:35:10Z"
}
```

NUNCA aceitar correção sem `reviewer_id` e `timestamp`. LGPD: `reviewer_id`
deve ser hash anonimizado, NUNCA email ou nome.

---

## 8. Pipeline — Regras de Roteamento

Pipelines disponíveis (em ordem de custo):

| Pipeline | Custo | Velocidade | Uso |
|---|---|---|---|
| YOLO v14/v15 | ~grátis | rápido | Base — toda detecção começa aqui |
| DINOv2 few-shot | baixo | rápido | Biblioteca pessoal do tenant |
| RT-DETR | médio | médio | Refino quando bbox suspeita |
| Template Matching | baixo | rápido | APENAS audit_mode |
| Claude Vision | alto | lento | Verificação semântica final |
| GPT/Gemini Vision | alto | lento | Fallback se Claude indisponível |

### Regras ALWAYS

- ALWAYS rodar YOLO como primeira passada
- ALWAYS rodar DINOv2 em paralelo se tenant tem biblioteca pessoal > 0
- ALWAYS persistir trilha de auditoria
- ALWAYS validar `model_version` no carregamento
- ALWAYS usar Confidence Router para decidir destino de detecção

### Regras NEVER

- NEVER usar Template Matching se classe está no YOLO (exceto `audit_mode=true`)
- NEVER chamar Claude/GPT/Gemini Vision sem passar pelo Confidence Router
- NEVER fazer auto-accept de classe sem threshold calibrado em
  `config/confidence-thresholds.json`
- NEVER misturar biblioteca pessoal de tenants no fine-tune do modelo base
- NEVER aceitar correção humana sem `reviewer_id` e `timestamp`
- NEVER inferir sem antes detectar a escala (ver Scale Detection Agent)
- NEVER subir threshold sem relatório de impacto
- NEVER deletar uma classe sem soft-delete (`deleted_at`)
- NEVER misturar detecções da legenda com detecções da planta

---

## 9. Custo e Performance

### Budget de Vision API

- Hard cap: **$0.50 por planta processada**
- Alerta em $0.30 (60% do cap)
- Se projeção > cap, Confidence Router rebaixa lowest-confidence para
  `send_to_review` em vez de `verify_with_vision`

### Latência alvo

| Operação | P50 | P95 |
|---|---|---|
| Render PDF (1 página A1) | 2s | 5s |
| YOLO + DINOv2 (paralelo) | 3s | 8s |
| Vision verification (lote) | 5s | 15s |
| Reconhecimento end-to-end | 15s | 45s |

---

## 10. Multi-tenancy — Regras de Isolamento

- Biblioteca pessoal: chave (`tenant_id`, `slug`)
- Embeddings DINOv2: índice pgvector particionado por `tenant_id`
- Telemetria: agregada por tenant; nunca expor entre tenants
- Fine-tune do modelo base: NUNCA inclui dados de tenant individual sem
  consentimento explícito
- MCP servers: autenticação obrigatória por `tenant_id`

---

## 11. Convenções de Código

### Stack
- Backend: Node.js + TypeScript
- Workers: BullMQ + Redis
- DB: Postgres + pgvector
- ML: Python (FastAPI) para inferência YOLO/DINOv2
- Frontend: SPA estática (ver `index.html`)

### Linguagem
- Código: inglês (variáveis, funções, comentários)
- Mensagens ao usuário: PT-BR
- Documentação interna: PT-BR
- Logs: inglês com `tenant_id` e `correlation_id`

### Estrutura de pastas
Ver §12.

### Nomenclatura
- Arquivos de agente: `*.agent.md`
- Arquivos de skill: `SKILL.md` dentro de pasta `skills/{nome}/`
- Hooks: `*.sh` em `.claude/hooks/`
- Migrations: `YYYYMMDDHHMM_descricao.sql`

---

## 12. Estrutura de Pastas

```
quantx/
├── CLAUDE.md                              ← este arquivo
├── .claude/
│   ├── settings.json                      ← registro de hooks
│   └── hooks/                             ← guardrails determinísticos
├── agents/                                ← subagents (L4)
│   ├── recognition/
│   ├── catalog/
│   └── quality/
├── skills/                                ← knowledge layer (L2)
├── plugins/                               ← MCP bundle (L5)
├── config/
│   ├── confidence-thresholds.json
│   ├── superclasses.json
│   └── pipeline-rules.json
└── index.html                             ← landing page (atual)
```

---

## 13. Testes Mínimos

Antes de qualquer deploy:
- Typecheck (`tsc --noEmit`) deve passar
- Smoke test do pipeline em planta de referência (`fixtures/plan-a1-1to100.pdf`)
- Validação de `classes.json` do modelo ativo vs DB

Comandos esperados (ainda não implementados — placeholder):
```bash
npm run typecheck
npm run test:smoke
npm run validate:model
```

---

## 14. Mudanças Sensíveis — Exigem Justificativa

As seguintes mudanças NUNCA devem ser feitas sem justificativa explícita
no commit message E sem rodar o hook correspondente:

| Mudança | Hook |
|---|---|
| Threshold de confiança | `pre-threshold-change.sh` |
| Troca de modelo YOLO ativo | `pre-model-swap.sh` |
| Adição de superclasse | revisão manual + atualização da taxonomia |
| Mudança no schema canônico | migration + validação retroativa |

---

## 15. O Que Está Implementado vs Planejado

**Implementado hoje:**
- Landing page (`index.html`)
- Branch de debug (`claude/debug-yolo-detection-rgaYp`)

**Planejado (ver proposta de arquitetura):**
- Camada 1: este `CLAUDE.md`
- Camada 2: 8 skills
- Camada 3: 5 hooks
- Camada 4: 7 subagents
- Camada 5: bundle MCP

Ordem de implementação: ver "Plano de Implementação em Fases" na proposta
técnica entregue em 2026-05-09.

---

## 16. Para o Claude Code — Diretrizes de Atuação

Ao trabalhar neste repositório:

1. SEMPRE leia este arquivo na primeira mensagem da sessão
2. Antes de editar qualquer arquivo em `config/`, leia o schema relacionado
3. Antes de adicionar uma nova classe, invoque o Catalog Review Agent
4. Antes de mudar threshold, gere relatório de impacto
5. Para perguntas sobre símbolos/normas, consulte §2 e §4 — não invente
6. Se for criar arquivo novo de agente/skill/hook, siga a estrutura de §12
7. Em caso de conflito entre instrução do usuário e regra deste arquivo,
   pergunte antes de prosseguir

---

_Última revisão: 2026-05-09. Versão: 1.0._
