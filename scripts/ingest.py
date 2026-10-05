"""
scripts/ingest.py

End-to-end ingestion pipeline:
  PDF files → text extraction → chunking → embeddings → PostgreSQL

Usage:
    python scripts/ingest.py --papers-dir data/papers/
    python scripts/ingest.py --papers-dir data/papers/ --reset
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from loguru import logger

from src.config import get_settings
from src.db.postgres import get_connection, init_db
from src.ingestion.chunker import chunk_document
from src.ingestion.embedder import embed_texts
from src.ingestion.pdf_loader import iter_papers_dir

settings = get_settings()


def upsert_paper(conn, filename: str, title: str, authors: list[str], num_pages: int) -> int:
    """Insert or update a paper record. Returns the paper id."""
    title_clean = (title or "").replace("\x00", "")
    authors_clean = [(a or "").replace("\x00", "") for a in (authors or [])]
    row = conn.execute(
        """
        INSERT INTO papers (filename, title, authors, num_pages)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (filename) DO UPDATE
            SET title = EXCLUDED.title,
                authors = EXCLUDED.authors,
                num_pages = EXCLUDED.num_pages,
                ingested_at = now()
        RETURNING id
        """,
        (filename, title_clean, authors_clean, num_pages),
    ).fetchone()
    return row[0]


def ingest_paper(conn, doc, reset: bool = False) -> int:
    """Ingest a single PaperDocument. Returns number of chunks inserted."""
    paper_id = upsert_paper(conn, doc.filename, doc.title, doc.authors, doc.num_pages)

    if reset:
        conn.execute("DELETE FROM chunks WHERE paper_id = %s", (paper_id,))

    # Check if already ingested (skip if chunks exist and not resetting)
    count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE paper_id = %s", (paper_id,)
    ).fetchone()[0]
    if count > 0 and not reset:
        logger.info(f"  '{doc.filename}' already ingested ({count} chunks). Use --reset to re-ingest.")
        return 0

    # Chunk
    chunks = chunk_document(doc)
    if not chunks:
        logger.warning(f"  No chunks generated for '{doc.filename}'")
        return 0

    # Embed
    texts = [c.content for c in chunks]
    embeddings: np.ndarray = embed_texts(texts, show_progress=len(chunks) > 50)

    # Insert chunks
    for chunk, embedding in zip(chunks, embeddings):
        content_clean = chunk.content.replace("\x00", "")
        conn.execute(
            """
            INSERT INTO chunks (paper_id, chunk_index, page_number, content, embedding)
            VALUES (%s, %s, %s, %s, %s::vector)
            ON CONFLICT (paper_id, chunk_index) DO UPDATE
                SET content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding
            """,
            (paper_id, chunk.chunk_index, chunk.page_number, content_clean, embedding.tolist()),
        )

    logger.success(f"  ✅ '{doc.filename}' → {len(chunks)} chunks ingested")
    return len(chunks)


def run_ingestion(papers_dir: Path, reset: bool = False) -> None:
    logger.info("=" * 55)
    logger.info("Research Intelligence Assistant — Ingestion Pipeline")
    logger.info("=" * 55)
    logger.info(f"Papers directory : {papers_dir}")
    logger.info(f"Reset mode       : {reset}")
    logger.info(f"Chunk size       : {settings.chunk_size} words")
    logger.info(f"Embedding model  : {settings.embedding_model}")
    logger.info("=" * 55)

    # Initialize DB schema
    init_db()

    pdf_files = list(papers_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {papers_dir}")
        sys.exit(1)

    # Ingest document by document (streamed, individual commit per paper)
    total_papers = 0
    total_chunks = 0
    for idx, doc in enumerate(iter_papers_dir(papers_dir), 1):
        logger.info(f"[{idx}/{len(pdf_files)}] Processing: '{doc.filename}' ({doc.num_pages} pages)")
        try:
            with get_connection() as conn:
                chunks_inserted = ingest_paper(conn, doc, reset=reset)
                total_chunks += chunks_inserted
                total_papers += 1
        except Exception as e:
            logger.error(f"Failed to ingest '{doc.filename}': {e}")

    logger.info("=" * 55)
    logger.success(f"Ingestion complete: {total_papers} papers processed, {total_chunks} chunks ingested")


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into the Research Intelligence Assistant")
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=Path("data/papers"),
        help="Directory containing PDF files (default: data/papers/)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing chunks and re-ingest all papers",
    )
    args = parser.parse_args()

    if not args.papers_dir.exists():
        logger.error(f"Directory not found: {args.papers_dir}")
        sys.exit(1)

    run_ingestion(args.papers_dir, reset=args.reset)


if __name__ == "__main__":
    main()
