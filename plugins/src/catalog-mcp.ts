/**
 * catalog-mcp — MCP server for managing the QuantX symbol catalog.
 *
 * Tools:
 *   catalog_list_classes     — list classes, filter by superclass / tenant_id
 *   catalog_get_class        — get one class by slug
 *   catalog_create_class     — create a new class (validates canonical schema)
 *   catalog_update_class     — update mutable fields (visual_description, invariances, nbr_reference)
 *   catalog_delete_class     — soft-delete (sets deleted_at)
 *   catalog_validate_schema  — validate a candidate class definition without persisting
 *
 * Storage: JSON file at CATALOG_PATH (default: config/catalog.json).
 * In production, swap readStore/writeStore for a Postgres-backed implementation.
 */

import { readFileSync, writeFileSync, existsSync } from 'fs';
import { resolve } from 'path';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  type Tool,
} from '@modelcontextprotocol/sdk/types.js';

// ── types ─────────────────────────────────────────────────────────────────────

interface ClassRecord {
  slug: string;
  className: string;
  superclass: string;
  visual_description: string;
  invariances: { rotation: boolean; flip: boolean; scale: boolean };
  reference_image_url: string;
  nbr_reference: string;
  tenant_id: string | null;
  created_at: string;
  version: number;
  deleted_at?: string;
}

interface CatalogStore {
  classes: ClassRecord[];
}

// ── storage ───────────────────────────────────────────────────────────────────

const CATALOG_PATH = process.env.CATALOG_PATH
  ?? resolve(process.cwd(), '../config/catalog.json');

function readStore(): CatalogStore {
  if (!existsSync(CATALOG_PATH)) return { classes: [] };
  return JSON.parse(readFileSync(CATALOG_PATH, 'utf-8')) as CatalogStore;
}

function writeStore(store: CatalogStore): void {
  writeFileSync(CATALOG_PATH, JSON.stringify(store, null, 2));
}

// ── validation ────────────────────────────────────────────────────────────────

const SLUG_RE = /^[a-z0-9]+(-[a-z0-9]+)*$/;
const CLASS_NAME_RE = /^[A-Z][A-Za-z0-9]{1,39}$/;
const VALID_SUPERCLASSES = new Set([
  'sprinkler/pendent', 'sprinkler/upright', 'sprinkler/sidewall',
  'sprinkler/concealed', 'sprinkler/dry',
  'extintor/pqs', 'extintor/co2', 'extintor/agua', 'extintor/affs', 'extintor/classe-k',
  'hidrante/parede', 'hidrante/recalque', 'hidrante/coluna-seca',
  'detector/fumaca', 'detector/temperatura', 'detector/multicriterio',
  'detector/gas', 'detector/beam',
  'alarme/manual', 'alarme/sirene', 'alarme/central', 'alarme/modulo',
  'valvula/governadora', 'valvula/retencao', 'valvula/gaveta', 'valvula/recalque',
  'bomba/principal', 'bomba/jockey', 'bomba/diesel',
  'mangotinho/parede',
  'iluminacao-emergencia/bloco', 'iluminacao-emergencia/pictograma-saida',
]);

function validateClassFields(c: Partial<ClassRecord>): string[] {
  const errors: string[] = [];
  if (!c.slug || !SLUG_RE.test(c.slug)) errors.push('slug must be kebab-case');
  if (!c.className || !CLASS_NAME_RE.test(c.className)) errors.push('className must be PascalCase ≤40 chars');
  if (!c.superclass || !VALID_SUPERCLASSES.has(c.superclass)) errors.push(`superclass '${c.superclass}' not in taxonomy`);
  const vd = c.visual_description ?? '';
  if (vd.length < 50 || vd.length > 300) errors.push('visual_description must be 50–300 chars');
  if (!c.nbr_reference) errors.push('nbr_reference is required');
  return errors;
}

// ── tools ─────────────────────────────────────────────────────────────────────

const TOOLS: Tool[] = [
  {
    name: 'catalog_list_classes',
    description: 'List symbol classes in the catalog. Optionally filter by superclass or tenant_id. Excludes soft-deleted by default.',
    inputSchema: {
      type: 'object',
      properties: {
        superclass:        { type: 'string', description: 'Filter by superclass (e.g. sprinkler/pendent)' },
        tenant_id:         { type: 'string', description: 'Filter by tenant. Use null for global classes.' },
        include_deleted:   { type: 'boolean', description: 'Include soft-deleted classes (default false)' },
      },
    },
  },
  {
    name: 'catalog_get_class',
    description: 'Get a single class definition by its slug.',
    inputSchema: {
      type: 'object',
      required: ['slug'],
      properties: {
        slug:      { type: 'string' },
        tenant_id: { type: 'string', description: 'Scope lookup to tenant (null = global)' },
      },
    },
  },
  {
    name: 'catalog_create_class',
    description: 'Create a new symbol class. Validates against the canonical schema. Fails if slug already exists for tenant.',
    inputSchema: {
      type: 'object',
      required: ['slug', 'className', 'superclass', 'visual_description', 'nbr_reference'],
      properties: {
        slug:                { type: 'string' },
        className:           { type: 'string' },
        superclass:          { type: 'string' },
        visual_description:  { type: 'string' },
        nbr_reference:       { type: 'string' },
        invariances:         { type: 'object', properties: { rotation: { type: 'boolean' }, flip: { type: 'boolean' }, scale: { type: 'boolean' } } },
        reference_image_url: { type: 'string' },
        tenant_id:           { type: 'string', description: 'null for global/base class' },
      },
    },
  },
  {
    name: 'catalog_update_class',
    description: 'Update mutable fields of an existing class (visual_description, invariances, nbr_reference). Increments version.',
    inputSchema: {
      type: 'object',
      required: ['slug'],
      properties: {
        slug:                { type: 'string' },
        tenant_id:           { type: 'string' },
        visual_description:  { type: 'string' },
        nbr_reference:       { type: 'string' },
        reference_image_url: { type: 'string' },
        invariances:         { type: 'object', properties: { rotation: { type: 'boolean' }, flip: { type: 'boolean' }, scale: { type: 'boolean' } } },
      },
    },
  },
  {
    name: 'catalog_delete_class',
    description: 'Soft-delete a class (sets deleted_at). Does NOT remove embeddings — call pgvector_delete_class_embeddings separately.',
    inputSchema: {
      type: 'object',
      required: ['slug'],
      properties: {
        slug:      { type: 'string' },
        tenant_id: { type: 'string' },
      },
    },
  },
  {
    name: 'catalog_validate_schema',
    description: 'Validate a candidate class definition against the canonical schema without persisting. Returns validation errors.',
    inputSchema: {
      type: 'object',
      required: ['slug', 'className', 'superclass', 'visual_description', 'nbr_reference'],
      properties: {
        slug:               { type: 'string' },
        className:          { type: 'string' },
        superclass:         { type: 'string' },
        visual_description: { type: 'string' },
        nbr_reference:      { type: 'string' },
      },
    },
  },
];

// ── handlers ──────────────────────────────────────────────────────────────────

function handleListClasses(args: Record<string, unknown>) {
  const { superclass, tenant_id, include_deleted = false } = args as {
    superclass?: string; tenant_id?: string; include_deleted?: boolean;
  };
  const store = readStore();
  let list = store.classes;
  if (!include_deleted) list = list.filter((c) => !c.deleted_at);
  if (superclass) list = list.filter((c) => c.superclass === superclass);
  if (tenant_id !== undefined) list = list.filter((c) => c.tenant_id === tenant_id);
  return { count: list.length, classes: list };
}

function handleGetClass(args: Record<string, unknown>) {
  const { slug, tenant_id } = args as { slug: string; tenant_id?: string };
  const store = readStore();
  const found = store.classes.find(
    (c) => c.slug === slug && (tenant_id === undefined || c.tenant_id === tenant_id),
  );
  if (!found) return { error: `class '${slug}' not found` };
  return found;
}

function handleCreateClass(args: Record<string, unknown>) {
  const input = args as Partial<ClassRecord>;
  const errors = validateClassFields(input);
  if (errors.length) return { error: 'validation_failed', details: errors };

  const store = readStore();
  const exists = store.classes.find(
    (c) => c.slug === input.slug && c.tenant_id === (input.tenant_id ?? null) && !c.deleted_at,
  );
  if (exists) return { error: `slug '${input.slug}' already exists for this tenant` };

  const record: ClassRecord = {
    slug: input.slug!,
    className: input.className!,
    superclass: input.superclass!,
    visual_description: input.visual_description!,
    invariances: input.invariances ?? { rotation: false, flip: false, scale: true },
    reference_image_url: input.reference_image_url ?? '',
    nbr_reference: input.nbr_reference!,
    tenant_id: input.tenant_id ?? null,
    created_at: new Date().toISOString(),
    version: 1,
  };
  store.classes.push(record);
  writeStore(store);
  return { created: true, class: record };
}

function handleUpdateClass(args: Record<string, unknown>) {
  const { slug, tenant_id, ...updates } = args as Record<string, unknown> & { slug: string; tenant_id?: string };
  const store = readStore();
  const idx = store.classes.findIndex(
    (c) => c.slug === slug && c.tenant_id === (tenant_id ?? null) && !c.deleted_at,
  );
  if (idx === -1) return { error: `class '${slug}' not found` };

  const record = store.classes[idx] as unknown as Record<string, unknown>;
  const allowed = ['visual_description', 'nbr_reference', 'reference_image_url', 'invariances'] as const;
  for (const key of allowed) {
    if (updates[key] !== undefined) record[key] = updates[key];
  }
  store.classes[idx].version += 1;
  writeStore(store);
  return { updated: true, class: store.classes[idx] };
}

function handleDeleteClass(args: Record<string, unknown>) {
  const { slug, tenant_id } = args as { slug: string; tenant_id?: string };
  const store = readStore();
  const idx = store.classes.findIndex(
    (c) => c.slug === slug && c.tenant_id === (tenant_id ?? null) && !c.deleted_at,
  );
  if (idx === -1) return { error: `class '${slug}' not found or already deleted` };
  store.classes[idx].deleted_at = new Date().toISOString();
  writeStore(store);
  return { deleted: true, slug, deleted_at: store.classes[idx].deleted_at };
}

function handleValidateSchema(args: Record<string, unknown>) {
  const errors = validateClassFields(args as Partial<ClassRecord>);
  return errors.length
    ? { valid: false, errors }
    : { valid: true, errors: [] };
}

// ── server bootstrap ──────────────────────────────────────────────────────────

const server = new Server(
  { name: 'quantx-catalog-mcp', version: '0.1.0' },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const { name, arguments: args = {} } = req.params;

  try {
    let result: unknown;
    switch (name) {
      case 'catalog_list_classes':    result = handleListClasses(args as Record<string, unknown>); break;
      case 'catalog_get_class':       result = handleGetClass(args as Record<string, unknown>); break;
      case 'catalog_create_class':    result = handleCreateClass(args as Record<string, unknown>); break;
      case 'catalog_update_class':    result = handleUpdateClass(args as Record<string, unknown>); break;
      case 'catalog_delete_class':    result = handleDeleteClass(args as Record<string, unknown>); break;
      case 'catalog_validate_schema': result = handleValidateSchema(args as Record<string, unknown>); break;
      default: return { content: [{ type: 'text', text: `Unknown tool: ${name}` }], isError: true };
    }
    return { content: [{ type: 'text', text: JSON.stringify(result, null, 2) }] };
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    return { content: [{ type: 'text', text: `Error: ${msg}` }], isError: true };
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
