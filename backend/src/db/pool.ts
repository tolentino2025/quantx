import pg from 'pg';

const { Pool } = pg;

// Supabase transaction pooler usa porta 6543, session pooler 5432.
// Para BullMQ workers (transações curtas) preferimos transaction mode (6543).
// DATABASE_URL pode conter ?pgbouncer=true para sinalizar isso.

const DATABASE_URL = process.env.DATABASE_URL;

if (!DATABASE_URL) {
  throw new Error('DATABASE_URL não definida — configure no .env ou nas variáveis de ambiente');
}

// Detecta se está apontando para Supabase (ajusta pool size e SSL)
const isSupabase = DATABASE_URL.includes('.supabase.co');
const isLocal    = DATABASE_URL.includes('localhost') || DATABASE_URL.includes('127.0.0.1');

export const pool = new Pool({
  connectionString: DATABASE_URL,
  max: isLocal ? 10 : 20,              // Supabase/RDS suportam mais conexões
  idleTimeoutMillis: 30_000,
  connectionTimeoutMillis: 5_000,
  ssl: isLocal ? false : { rejectUnauthorized: false },
});

pool.on('error', (err) => {
  console.error('[db] idle client error:', err.message);
});

export async function dbHealthCheck(): Promise<boolean> {
  try {
    const client = await pool.connect();
    await client.query('SELECT 1');
    client.release();
    return true;
  } catch {
    return false;
  }
}
