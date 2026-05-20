import pg from 'pg';

const { Pool } = pg;

// Supabase transaction pooler usa porta 6543, session pooler 5432.
// Para BullMQ workers (transações curtas) preferimos transaction mode (6543).
// DATABASE_URL pode conter ?pgbouncer=true para sinalizar isso.

let _pool: pg.Pool | null = null;

function getPool(): pg.Pool {
  if (_pool) return _pool;

  const DATABASE_URL = process.env.DATABASE_URL;
  if (!DATABASE_URL) {
    throw new Error('DATABASE_URL não definida — configure no .env ou nas variáveis de ambiente');
  }

  const isLocal = DATABASE_URL.includes('localhost') || DATABASE_URL.includes('127.0.0.1');

  _pool = new Pool({
    connectionString: DATABASE_URL,
    max: isLocal ? 10 : 20,              // Supabase/RDS suportam mais conexões
    idleTimeoutMillis: 30_000,
    connectionTimeoutMillis: 5_000,
    ssl: isLocal ? false : { rejectUnauthorized: false },
  });

  _pool.on('error', (err) => {
    console.error('[db] idle client error:', err.message);
  });

  return _pool;
}

// Lazy proxy — o Pool só é criado no primeiro uso, não no import do módulo.
export const pool: pg.Pool = new Proxy({} as pg.Pool, {
  get(_target, prop) {
    const p = getPool();
    const value = (p as unknown as Record<string | symbol, unknown>)[prop];
    return typeof value === 'function' ? (value as (...a: unknown[]) => unknown).bind(p) : value;
  },
});

export async function dbHealthCheck(): Promise<boolean> {
  try {
    const client = await getPool().connect();
    await client.query('SELECT 1');
    client.release();
    return true;
  } catch {
    return false;
  }
}
