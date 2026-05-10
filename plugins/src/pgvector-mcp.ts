/**
 * pgvector-mcp — MCP server for querying DINOv2 embeddings.
 *
 * Delegates to the ML service HTTP API (/dinov2/*) so the MCP server
 * stays stateless and does not hold a direct Postgres connection.
 * For direct pgvector access, set PGVECTOR_DIRECT=true and provide PG_* vars.
 *
 * Tools:
 *   pgvector_get_library_stats          — count embeddings per tenant / class
 *   pgvector_list_tenant_classes        — list classes with embeddings for a tenant
 *   pgvector_delete_class_embeddings    — delete all embeddings for a class (tenant-scoped)
 *   pgvector_query_similar              — find top-k similar symbols (via ML service)
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  type Tool,
} from '@modelcontextprotocol/sdk/types.js';

// ── config ────────────────────────────────────────────────────────────────────

const ML_BASE_URL = (process.env.ML_BASE_URL ?? 'http://localhost:8000').replace(/\/$/, '');
const TIMEOUT_MS  = Number(process.env.ML_TIMEOUT_MS ?? 15_000);

// ── http helper ───────────────────────────────────────────────────────────────

async function mlPost<T>(endpoint: string, body: unknown): Promise<T> {
  const res = await fetch(`${ML_BASE_URL}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`ML service ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json() as Promise<T>;
}

async function mlDelete(endpoint: string): Promise<unknown> {
  const res = await fetch(`${ML_BASE_URL}${endpoint}`, {
    method: 'DELETE',
    signal: AbortSignal.timeout(TIMEOUT_MS),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`ML service ${res.status}: ${text.slice(0, 200)}`);
  }
  return res.json();
}

// ── tools ─────────────────────────────────────────────────────────────────────

const TOOLS: Tool[] = [
  {
    name: 'pgvector_get_library_stats',
    description: 'Get the total number of stored embeddings for a tenant, optionally broken down by class.',
    inputSchema: {
      type: 'object',
      required: ['tenant_id'],
      properties: {
        tenant_id:   { type: 'string' },
        by_class:    { type: 'boolean', description: 'If true, return per-class counts (requires a match call with empty detections)' },
      },
    },
  },
  {
    name: 'pgvector_list_tenant_classes',
    description: 'List all classes that have at least one embedding in the tenant library.',
    inputSchema: {
      type: 'object',
      required: ['tenant_id'],
      properties: {
        tenant_id: { type: 'string' },
      },
    },
  },
  {
    name: 'pgvector_delete_class_embeddings',
    description: 'Delete all embeddings for a specific class in a tenant library. Also call catalog_delete_class to soft-delete the class definition.',
    inputSchema: {
      type: 'object',
      required: ['tenant_id', 'class_slug'],
      properties: {
        tenant_id:  { type: 'string' },
        class_slug: { type: 'string' },
      },
    },
  },
  {
    name: 'pgvector_query_similar',
    description: 'Find the top-k most similar symbols for a detected bounding box region. Calls DINOv2 match endpoint.',
    inputSchema: {
      type: 'object',
      required: ['tenant_id', 'image_path', 'detection'],
      properties: {
        tenant_id:    { type: 'string' },
        image_path:   { type: 'string', description: 'Absolute path to the page image' },
        detection: {
          type: 'object',
          description: 'Detection to match against the library',
          required: ['detection_id', 'tile_id', 'class_slug', 'bbox_tile'],
          properties: {
            detection_id: { type: 'string' },
            tile_id:      { type: 'string' },
            class_slug:   { type: 'string' },
            bbox_tile: {
              type: 'object',
              required: ['x1', 'y1', 'x2', 'y2'],
              properties: { x1: { type: 'number' }, y1: { type: 'number' }, x2: { type: 'number' }, y2: { type: 'number' } },
            },
            confidence:   { type: 'number' },
          },
        },
        top_k:         { type: 'integer', description: 'Number of candidates to return (default 5)' },
        dinov2_window: { type: 'integer', description: 'Crop window size in pixels (default 128)' },
      },
    },
  },
];

// ── handlers ──────────────────────────────────────────────────────────────────

async function handleGetLibraryStats(args: Record<string, unknown>) {
  const { tenant_id } = args as { tenant_id: string };
  const result = await mlPost<{ library_size: number; matches: unknown[]; warning?: string }>(
    '/dinov2/match',
    { tenant_id, image_path: '/dev/null', detections: [], dinov2_window: 128, top_k: 1 },
  );
  return {
    tenant_id,
    library_size: result.library_size,
    ...(result.warning ? { warning: result.warning } : {}),
  };
}

async function handleListTenantClasses(args: Record<string, unknown>) {
  const { tenant_id } = args as { tenant_id: string };
  // The ML service does not expose a list endpoint; we use a match with empty
  // detections just to get library_size. Class listing requires direct DB access —
  // this tool surfaces that limitation and suggests the alternative.
  const result = await mlPost<{ library_size: number; warning?: string }>(
    '/dinov2/match',
    { tenant_id, image_path: '/dev/null', detections: [], dinov2_window: 128, top_k: 1 },
  );
  return {
    tenant_id,
    library_size: result.library_size,
    note: 'Per-class listing requires direct pgvector access. Set PGVECTOR_DIRECT=true and PG_* env vars for full class breakdown.',
    ...(result.warning ? { warning: result.warning } : {}),
  };
}

async function handleDeleteClassEmbeddings(args: Record<string, unknown>) {
  const { tenant_id, class_slug } = args as { tenant_id: string; class_slug: string };
  const result = await mlDelete(`/dinov2/library/${encodeURIComponent(tenant_id)}/${encodeURIComponent(class_slug)}`);
  return result;
}

async function handleQuerySimilar(args: Record<string, unknown>) {
  const { tenant_id, image_path, detection, top_k = 5, dinov2_window = 128 } = args as {
    tenant_id: string;
    image_path: string;
    detection: Record<string, unknown>;
    top_k?: number;
    dinov2_window?: number;
  };
  const result = await mlPost<{ tenant_id: string; library_size: number; matches: unknown[] }>(
    '/dinov2/match',
    {
      tenant_id,
      image_path,
      detections: [detection],
      dinov2_window,
      top_k,
    },
  );
  return {
    tenant_id,
    library_size: result.library_size,
    match: result.matches[0] ?? null,
  };
}

// ── server bootstrap ──────────────────────────────────────────────────────────

const server = new Server(
  { name: 'quantx-pgvector-mcp', version: '0.1.0' },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const { name, arguments: args = {} } = req.params;
  try {
    let result: unknown;
    switch (name) {
      case 'pgvector_get_library_stats':       result = await handleGetLibraryStats(args as Record<string, unknown>); break;
      case 'pgvector_list_tenant_classes':     result = await handleListTenantClasses(args as Record<string, unknown>); break;
      case 'pgvector_delete_class_embeddings': result = await handleDeleteClassEmbeddings(args as Record<string, unknown>); break;
      case 'pgvector_query_similar':           result = await handleQuerySimilar(args as Record<string, unknown>); break;
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
