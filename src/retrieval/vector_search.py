"""
Vector similarity search using pgvector.
"""

from dataclasses import dataclass

import numpy as np
from loguru import logger

from src.config import get_settings
from src.db.postgres import get_connection
from src.ingestion.embedder import embed_query

settings = get_settings()


@dataclass
class SearchResult:
    chunk_id: int
    paper_id: int
    paper_title: str
    paper_filename: str
    chunk_index: int
    page_number: int
    content: str
    score: float       # cosine similarity (0–1, higher = more similar)


def vector_search(
    query: str,
    top_k: int | None = None,
    paper_filter: list[str] | None = None,   # optional: restrict to specific filenames
) -> list[SearchResult]:
    """
    Embed the query and retrieve the top-k most similar chunks from pgvector.

    Args:
        query:        Natural language question.
        top_k:        Number of results to return (default from config).
        paper_filter: If provided, only search within these paper filenames.

    Returns:
        List of SearchResult sorted by descending similarity score.
    """
    top_k = top_k or settings.retrieval_top_k
    query_embedding = embed_query(query)

    # pgvector uses <=> for cosine distance (0 = identical, 2 = opposite)
    # score = 1 - cosine_distance  (so 1 = identical)
    base_sql = """
        SELECT
            c.id            AS chunk_id,
            c.paper_id,
            p.title         AS paper_title,
            p.filename      AS paper_filename,
            c.chunk_index,
            c.page_number,
            c.content,
            1 - (c.embedding <=> %s::vector) AS score
        FROM chunks c
        JOIN papers p ON p.id = c.paper_id
        {where_clause}
        ORDER BY score DESC
        LIMIT %s
    """

    if paper_filter:
        where_clause = "WHERE p.filename = ANY(%s)"
    else:
        where_clause = ""

    sql = base_sql.format(where_clause=where_clause)

    with get_connection() as conn:
        if paper_filter:
            rows = conn.execute(sql, (query_embedding, paper_filter, top_k)).fetchall()
        else:
            rows = conn.execute(sql, (query_embedding, top_k)).fetchall()

    results = [
        SearchResult(
            chunk_id=row[0],
            paper_id=row[1],
            paper_title=row[2],
            paper_filename=row[3],
            chunk_index=row[4],
            page_number=row[5],
            content=row[6],
            score=float(row[7]),
        )
        for row in rows
    ]

    logger.debug(f"Vector search: '{query[:60]}...' → {len(results)} results")
    return results


def format_chunks_for_prompt(results: list[SearchResult]) -> str:
    """
    Format search results as a numbered list for inclusion in LLM prompts.
    """
    parts = []
    for i, r in enumerate(results, 1):
        parts.append(
            f"[{i}] Source: {r.paper_title} (p. {r.page_number})\n"
            f"{r.content}"
        )
    return "\n\n---\n\n".join(parts)
