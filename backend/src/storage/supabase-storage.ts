import { createClient, type SupabaseClient } from '@supabase/supabase-js';

export const BUCKETS = {
  PLANS_ORIGINAL:    'plans-original',
  PLANS_PAGES:       'plans-pages',
  PLANS_PROCESSED:   'plans-processed',
  SYMBOLS_LIBRARY:   'symbols-library',
  TRAINING_DATASETS: 'training-datasets',
  EXPORTS:           'exports',
} as const;

export type BucketName = typeof BUCKETS[keyof typeof BUCKETS];

function getClient(): SupabaseClient {
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) {
    throw new Error('SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set');
  }
  return createClient(url, key, { auth: { persistSession: false } });
}

// ── upload ────────────────────────────────────────────────────────────────────

export async function uploadBuffer(
  bucket: BucketName,
  storagePath: string,
  data: Buffer,
  contentType = 'application/octet-stream',
): Promise<void> {
  const { error } = await getClient()
    .storage.from(bucket)
    .upload(storagePath, data, { contentType, upsert: true });
  if (error) throw new Error(`Storage upload failed [${bucket}/${storagePath}]: ${error.message}`);
}

// ── download ──────────────────────────────────────────────────────────────────

export async function downloadBuffer(bucket: BucketName, storagePath: string): Promise<Buffer> {
  const { data, error } = await getClient().storage.from(bucket).download(storagePath);
  if (error) throw new Error(`Storage download failed [${bucket}/${storagePath}]: ${error.message}`);
  return Buffer.from(await (data as Blob).arrayBuffer());
}

// ── signed URL ────────────────────────────────────────────────────────────────

export async function getSignedUrl(
  bucket: BucketName,
  storagePath: string,
  expiresInSeconds = 3_600,
): Promise<string> {
  const { data, error } = await getClient()
    .storage.from(bucket)
    .createSignedUrl(storagePath, expiresInSeconds);
  if (error) throw new Error(`Signed URL failed [${bucket}/${storagePath}]: ${error.message}`);
  return data.signedUrl;
}

// ── bucket bootstrap ──────────────────────────────────────────────────────────

export async function ensureBuckets(): Promise<{ created: string[]; existing: string[] }> {
  const client = getClient();
  const { data: list, error } = await client.storage.listBuckets();
  if (error) throw new Error(`Cannot list buckets: ${error.message}`);

  const existingNames = new Set(list.map((b) => b.name));
  const created: string[] = [];
  const existing: string[] = [];

  for (const bucket of Object.values(BUCKETS)) {
    if (existingNames.has(bucket)) {
      existing.push(bucket);
    } else {
      const { error: ce } = await client.storage.createBucket(bucket, {
        public: false,
        fileSizeLimit: 500 * 1024 * 1024,
      });
      if (ce && !ce.message.toLowerCase().includes('already exists')) {
        throw new Error(`Failed to create bucket "${bucket}": ${ce.message}`);
      }
      created.push(bucket);
    }
  }

  return { created, existing };
}
