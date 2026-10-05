"""
Database layer for PostgreSQL + pgvector.
Handles connection pooling, schema creation, and helper queries.
"""

from contextlib import contextmanager
from typing import Generator

import psycopg
from loguru import logger
from pgvector.psycopg import register_vector

from src.config import get_settings

settings = get_settings()

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS papers (
    id          SERIAL PRIMARY KEY,
    title       TEXT,
    authors     TEXT[],
    filename    TEXT UNIQUE NOT NULL,
    abstract    TEXT,
    num_pages   INT,
    ingested_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    paper_id    INT NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    page_number INT,
    content     TEXT NOT NULL,
    embedding   vector({dim}),
    UNIQUE(paper_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _conn_str() -> str:
    s = settings
    return (
        f"host={s.postgres_host} port={s.postgres_port} "
        f"dbname={s.postgres_db} user={s.postgres_user} "
        f"password={s.postgres_password}"
    )


@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    """Yield a raw psycopg connection with pgvector registered."""
    conn = psycopg.connect(_conn_str())
    register_vector(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables and indexes if they don't exist yet."""
    ddl = DDL.format(dim=settings.embedding_dimension)
    with get_connection() as conn:
        conn.execute(ddl)
    logger.success("PostgreSQL schema initialized ✅")
