/**
 * telemetry-mcp — MCP server for querying recognition telemetry.
 *
 * Reads JSONL files written by RecognitionPipeline._writeTelemetry()
 * from TELEMETRY_DIR (default: /tmp/quantx-telemetry/).
 *
 * Tools:
 *   telemetry_get_stats         — aggregate stats for a date range
 *   telemetry_get_plan_history  — all runs recorded for a specific plan_id
 *   telemetry_get_tenant_summary — detection totals and decision breakdown by tenant
 *   telemetry_list_recent_runs  — latest N runs across all plans
 */

import { readdirSync, readFileSync, existsSync } from 'fs';
import { join } from 'path';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  type Tool,
} from '@modelcontextprotocol/sdk/types.js';

// ── types ─────────────────────────────────────────────────────────────────────

interface TelemetryEntry {
  timestamp: string;
  plan_id: string;
  page_number: number;
  tenant_id: string;
  scale: string;
  model_version: string;
  detection_count: number;
  decisions: {
    auto_accept: number;
    verify_with_vision: number;
    send_to_review: number;
    rejected: number;
  };
}

// ── reader ────────────────────────────────────────────────────────────────────

const TELEMETRY_DIR = process.env.TELEMETRY_DIR ?? '/tmp/quantx-telemetry';

function loadEntries(fromDate?: string, toDate?: string): TelemetryEntry[] {
  if (!existsSync(TELEMETRY_DIR)) return [];

  const files = readdirSync(TELEMETRY_DIR)
    .filter((f) => f.endsWith('_recognition.jsonl'))
    .sort();

  const entries: TelemetryEntry[] = [];
  for (const file of files) {
    const dateStr = file.slice(0, 8); // YYYYMMDD
    if (fromDate && dateStr < fromDate.replace(/-/g, '')) continue;
    if (toDate   && dateStr > toDate.replace(/-/g, ''))   continue;

    const lines = readFileSync(join(TELEMETRY_DIR, file), 'utf-8').trim().split('\n').filter(Boolean);
    for (const line of lines) {
      try { entries.push(JSON.parse(line) as TelemetryEntry); } catch { /* skip malformed */ }
    }
  }
  return entries;
}

// ── tools ─────────────────────────────────────────────────────────────────────

const TOOLS: Tool[] = [
  {
    name: 'telemetry_get_stats',
    description: 'Aggregate recognition statistics for a date range. Returns totals and decision breakdown.',
    inputSchema: {
      type: 'object',
      properties: {
        from_date:  { type: 'string', description: 'ISO date YYYY-MM-DD (inclusive)' },
        to_date:    { type: 'string', description: 'ISO date YYYY-MM-DD (inclusive)' },
        tenant_id:  { type: 'string', description: 'Filter by tenant' },
      },
    },
  },
  {
    name: 'telemetry_get_plan_history',
    description: 'Get all recorded runs for a specific plan_id, sorted by timestamp.',
    inputSchema: {
      type: 'object',
      required: ['plan_id'],
      properties: {
        plan_id: { type: 'string' },
      },
    },
  },
  {
    name: 'telemetry_get_tenant_summary',
    description: 'Summarise detection totals, decision rates and active model versions for each tenant.',
    inputSchema: {
      type: 'object',
      properties: {
        from_date: { type: 'string' },
        to_date:   { type: 'string' },
      },
    },
  },
  {
    name: 'telemetry_list_recent_runs',
    description: 'List the most recent recognition runs across all plans.',
    inputSchema: {
      type: 'object',
      properties: {
        limit:     { type: 'integer', description: 'Max runs to return (default 20)' },
        tenant_id: { type: 'string', description: 'Filter by tenant' },
      },
    },
  },
];

// ── handlers ──────────────────────────────────────────────────────────────────

function handleGetStats(args: Record<string, unknown>) {
  const { from_date, to_date, tenant_id } = args as {
    from_date?: string; to_date?: string; tenant_id?: string;
  };

  let entries = loadEntries(from_date, to_date);
  if (tenant_id) entries = entries.filter((e) => e.tenant_id === tenant_id);

  const totals = { runs: 0, detections: 0, auto_accept: 0, verify_with_vision: 0, send_to_review: 0, rejected: 0 };
  const byDate: Record<string, typeof totals> = {};

  for (const e of entries) {
    const date = e.timestamp.slice(0, 10);
    if (!byDate[date]) byDate[date] = { ...totals };
    for (const day of [totals, byDate[date]]) {
      day.runs            += 1;
      day.detections      += e.detection_count;
      day.auto_accept     += e.decisions.auto_accept;
      day.verify_with_vision += e.decisions.verify_with_vision;
      day.send_to_review  += e.decisions.send_to_review;
      day.rejected        += e.decisions.rejected;
    }
  }

  const autoAcceptRate = totals.detections > 0
    ? (totals.auto_accept / totals.detections).toFixed(3)
    : '0';

  return { totals, auto_accept_rate: autoAcceptRate, by_date: byDate };
}

function handleGetPlanHistory(args: Record<string, unknown>) {
  const { plan_id } = args as { plan_id: string };
  const entries = loadEntries()
    .filter((e) => e.plan_id === plan_id)
    .sort((a, b) => a.timestamp.localeCompare(b.timestamp));
  return { plan_id, run_count: entries.length, runs: entries };
}

function handleGetTenantSummary(args: Record<string, unknown>) {
  const { from_date, to_date } = args as { from_date?: string; to_date?: string };
  const entries = loadEntries(from_date, to_date);

  const tenants: Record<string, {
    runs: number; detections: number;
    decisions: { auto_accept: number; verify_with_vision: number; send_to_review: number; rejected: number };
    model_versions: Set<string>;
    scales: Record<string, number>;
  }> = {};

  for (const e of entries) {
    if (!tenants[e.tenant_id]) {
      tenants[e.tenant_id] = {
        runs: 0, detections: 0,
        decisions: { auto_accept: 0, verify_with_vision: 0, send_to_review: 0, rejected: 0 },
        model_versions: new Set(),
        scales: {},
      };
    }
    const t = tenants[e.tenant_id];
    t.runs            += 1;
    t.detections      += e.detection_count;
    t.decisions.auto_accept        += e.decisions.auto_accept;
    t.decisions.verify_with_vision += e.decisions.verify_with_vision;
    t.decisions.send_to_review     += e.decisions.send_to_review;
    t.decisions.rejected           += e.decisions.rejected;
    t.model_versions.add(e.model_version);
    t.scales[e.scale] = (t.scales[e.scale] ?? 0) + 1;
  }

  return Object.fromEntries(
    Object.entries(tenants).map(([tid, t]) => [
      tid,
      { ...t, model_versions: [...t.model_versions] },
    ]),
  );
}

function handleListRecentRuns(args: Record<string, unknown>) {
  const { limit = 20, tenant_id } = args as { limit?: number; tenant_id?: string };
  let entries = loadEntries().sort((a, b) => b.timestamp.localeCompare(a.timestamp));
  if (tenant_id) entries = entries.filter((e) => e.tenant_id === tenant_id);
  return entries.slice(0, limit);
}

// ── server bootstrap ──────────────────────────────────────────────────────────

const server = new Server(
  { name: 'quantx-telemetry-mcp', version: '0.1.0' },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const { name, arguments: args = {} } = req.params;
  try {
    let result: unknown;
    switch (name) {
      case 'telemetry_get_stats':          result = handleGetStats(args as Record<string, unknown>); break;
      case 'telemetry_get_plan_history':   result = handleGetPlanHistory(args as Record<string, unknown>); break;
      case 'telemetry_get_tenant_summary': result = handleGetTenantSummary(args as Record<string, unknown>); break;
      case 'telemetry_list_recent_runs':   result = handleListRecentRuns(args as Record<string, unknown>); break;
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
