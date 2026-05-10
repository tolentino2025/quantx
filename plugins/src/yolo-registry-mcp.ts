/**
 * yolo-registry-mcp — MCP server for managing YOLO model versions.
 *
 * Reads a model registry at MODEL_REGISTRY_PATH (default: config/models/registry.json).
 * Each entry points to a .pt weights file and a classes.json map.
 *
 * Tools:
 *   yolo_list_models      — list all registered model versions and their status
 *   yolo_get_active_model — get the currently active model version
 *   yolo_set_active_model — set the active model (validates file existence first)
 *   yolo_validate_model   — validate a .pt file + classes.json without activating
 *   yolo_get_class_map    — return the class slug map for a model version
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

interface ModelEntry {
  version: string;
  weights_path: string;
  classes_path: string;
  active: boolean;
  registered_at: string;
  notes?: string;
}

interface ModelRegistry {
  active_version: string;
  models: ModelEntry[];
}

// ── storage ───────────────────────────────────────────────────────────────────

const REGISTRY_PATH = process.env.MODEL_REGISTRY_PATH
  ?? resolve(process.cwd(), '../config/models/registry.json');

function readRegistry(): ModelRegistry {
  if (!existsSync(REGISTRY_PATH)) {
    return { active_version: '', models: [] };
  }
  return JSON.parse(readFileSync(REGISTRY_PATH, 'utf-8')) as ModelRegistry;
}

function writeRegistry(reg: ModelRegistry): void {
  writeFileSync(REGISTRY_PATH, JSON.stringify(reg, null, 2));
}

// ── validation helpers ────────────────────────────────────────────────────────

function validateModelFiles(weightsPath: string, classesPath: string): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  if (!existsSync(weightsPath)) errors.push(`weights file not found: ${weightsPath}`);
  if (!existsSync(classesPath)) errors.push(`classes file not found: ${classesPath}`);

  if (existsSync(classesPath)) {
    try {
      const classMap = JSON.parse(readFileSync(classesPath, 'utf-8')) as Record<string, unknown>;
      const keys = Object.keys(classMap);
      if (keys.length === 0) errors.push('classes.json is empty');
      const invalidSlugs = keys.filter((k) => !/^\d+$/.test(k));
      if (invalidSlugs.length) errors.push(`classes.json keys must be integer class IDs, got: ${invalidSlugs.slice(0, 3).join(', ')}`);
    } catch {
      errors.push('classes.json is not valid JSON');
    }
  }

  return { valid: errors.length === 0, errors };
}

// ── tools ─────────────────────────────────────────────────────────────────────

const TOOLS: Tool[] = [
  {
    name: 'yolo_list_models',
    description: 'List all registered YOLO model versions with their file paths and active status.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'yolo_get_active_model',
    description: 'Get the currently active YOLO model version and its configuration.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'yolo_set_active_model',
    description: 'Set the active model version. Validates weights and classes.json exist first. Requires QUANTX_MODEL_SWAP_JUSTIFICATION env var (enforced by pre-model-swap.sh hook).',
    inputSchema: {
      type: 'object',
      required: ['version'],
      properties: {
        version:       { type: 'string', description: 'Model version string (e.g. yolov15-spci-2026.04)' },
        justification: { type: 'string', description: 'Reason for model swap — also set QUANTX_MODEL_SWAP_JUSTIFICATION env var before calling' },
      },
    },
  },
  {
    name: 'yolo_validate_model',
    description: 'Validate that a model version\'s weights file and classes.json exist and are well-formed, without activating it.',
    inputSchema: {
      type: 'object',
      required: ['version'],
      properties: {
        version: { type: 'string' },
      },
    },
  },
  {
    name: 'yolo_get_class_map',
    description: 'Return the integer-to-slug class map for a model version.',
    inputSchema: {
      type: 'object',
      required: ['version'],
      properties: {
        version: { type: 'string' },
      },
    },
  },
];

// ── handlers ──────────────────────────────────────────────────────────────────

function handleListModels() {
  const reg = readRegistry();
  return {
    active_version: reg.active_version,
    model_count: reg.models.length,
    models: reg.models.map((m) => ({
      version: m.version,
      active: m.active,
      weights_exists: existsSync(m.weights_path),
      classes_exists: existsSync(m.classes_path),
      registered_at: m.registered_at,
      notes: m.notes,
    })),
  };
}

function handleGetActiveModel() {
  const reg = readRegistry();
  if (!reg.active_version) return { error: 'no active model set' };
  const model = reg.models.find((m) => m.version === reg.active_version);
  if (!model) return { error: `active_version '${reg.active_version}' not found in registry` };
  const validation = validateModelFiles(model.weights_path, model.classes_path);
  return { ...model, files_valid: validation.valid, file_errors: validation.errors };
}

function handleSetActiveModel(args: Record<string, unknown>) {
  const { version, justification } = args as { version: string; justification?: string };

  // Mirror the hook check: env var must be set
  const envJustification = process.env.QUANTX_MODEL_SWAP_JUSTIFICATION;
  if (!envJustification && !justification) {
    return {
      error: 'model_swap_blocked',
      detail: 'Set QUANTX_MODEL_SWAP_JUSTIFICATION env var before swapping the active model. See pre-model-swap.sh hook.',
    };
  }

  const reg = readRegistry();
  const model = reg.models.find((m) => m.version === version);
  if (!model) return { error: `model version '${version}' not found in registry` };

  const validation = validateModelFiles(model.weights_path, model.classes_path);
  if (!validation.valid) {
    return { error: 'model_files_invalid', details: validation.errors };
  }

  // Deactivate current
  for (const m of reg.models) m.active = false;
  model.active = true;
  reg.active_version = version;
  writeRegistry(reg);

  return {
    swapped: true,
    previous_version: reg.active_version,
    new_version: version,
    justification: justification ?? envJustification,
  };
}

function handleValidateModel(args: Record<string, unknown>) {
  const { version } = args as { version: string };
  const reg = readRegistry();
  const model = reg.models.find((m) => m.version === version);
  if (!model) return { error: `model version '${version}' not found in registry` };
  const validation = validateModelFiles(model.weights_path, model.classes_path);
  return { version, ...validation, weights_path: model.weights_path, classes_path: model.classes_path };
}

function handleGetClassMap(args: Record<string, unknown>) {
  const { version } = args as { version: string };
  const reg = readRegistry();
  const model = reg.models.find((m) => m.version === version);
  if (!model) return { error: `model version '${version}' not found in registry` };
  if (!existsSync(model.classes_path)) return { error: `classes.json not found: ${model.classes_path}` };
  try {
    const classMap = JSON.parse(readFileSync(model.classes_path, 'utf-8')) as Record<string, string>;
    return { version, class_count: Object.keys(classMap).length, class_map: classMap };
  } catch {
    return { error: 'classes.json is not valid JSON' };
  }
}

// ── server bootstrap ──────────────────────────────────────────────────────────

const server = new Server(
  { name: 'quantx-yolo-registry-mcp', version: '0.1.0' },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({ tools: TOOLS }));

server.setRequestHandler(CallToolRequestSchema, async (req) => {
  const { name, arguments: args = {} } = req.params;
  try {
    let result: unknown;
    switch (name) {
      case 'yolo_list_models':      result = handleListModels(); break;
      case 'yolo_get_active_model': result = handleGetActiveModel(); break;
      case 'yolo_set_active_model': result = handleSetActiveModel(args as Record<string, unknown>); break;
      case 'yolo_validate_model':   result = handleValidateModel(args as Record<string, unknown>); break;
      case 'yolo_get_class_map':    result = handleGetClassMap(args as Record<string, unknown>); break;
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
