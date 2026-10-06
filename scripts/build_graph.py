"""
scripts/build_graph.py

Extracts entities and relationships from papers and builds the Knowledge Graph in Neo4j.

Usage:
    python scripts/build_graph.py
    python scripts/build_graph.py --papers-dir data/papers/
    python scripts/build_graph.py --from-db
    python scripts/build_graph.py --reset
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger

from src.db.neo4j_client import get_neo4j_client
from src.db.postgres import get_connection
from src.graph.builder import GraphBuilder
from src.graph.extractor import extract_graph_elements
from src.ingestion.pdf_loader import load_papers_dir
from src.llm.factory import get_llm_client


def clear_graph() -> None:
    """Clear all nodes and relationships in Neo4j."""
    client = get_neo4j_client()
    logger.warning("Clearing Neo4j knowledge graph...")
    client.run("MATCH (n) DETACH DELETE n")
    client.init_constraints()
    logger.success("Neo4j graph cleared and constraints initialized ✅")


def is_paper_in_graph(client, filename: str) -> bool:
    """Check if paper is already processed with relationships in Neo4j."""
    res = client.run("MATCH (p:Paper {filename: $filename})-[r]->() RETURN count(r) AS c", filename=filename)
    return bool(res and res[0]["c"] > 0)


def build_from_database(builder: GraphBuilder, llm, limit: int | None = None) -> None:
    """Build knowledge graph from papers already stored in PostgreSQL."""
    logger.info("Fetching papers from PostgreSQL...")
    with get_connection() as conn:
        query = "SELECT id, filename, title, authors FROM papers ORDER BY id"
        if limit:
            query += f" LIMIT {limit}"
        papers = conn.execute(query).fetchall()

    if not papers:
        logger.warning("No papers found in PostgreSQL database. Ingest papers first.")
        return

    total = len(papers)
    logger.info(f"Processing up to {total} paper(s) for graph construction...")

    import time
    for idx, (p_id, filename, title, authors) in enumerate(papers, 1):
        if is_paper_in_graph(builder.client, filename):
            logger.info(f"[{idx}/{total}] Skipping '{filename}': already extracted in Neo4j graph")
            continue

        start_time = time.time()
        logger.info(f"[{idx}/{total}] Extracting entities from '{filename}' ('{(title or '')[:40]}')...")

        with get_connection() as conn:
            # First 4 chunks usually contain title, abstract, and introduction with key concepts
            chunks = conn.execute(
                "SELECT content FROM chunks WHERE paper_id = %s ORDER BY chunk_index LIMIT 4",
                (p_id,),
            ).fetchall()

        paper_text = "\n\n".join(c[0] for c in chunks)
        if not paper_text.strip():
            logger.warning(f"No chunk text found for {filename}")
            continue

        extraction = extract_graph_elements(
            llm=llm,
            paper_title=title or filename,
            paper_filename=filename,
            text=paper_text,
            max_text_chars=4000,
        )
        builder.build_from_extraction(extraction, authors=authors or [])
        elapsed = time.time() - start_time
        logger.success(f"[{idx}/{total}] ✅ '{filename}' finished in {elapsed:.1f}s")


def build_from_directory(papers_dir: Path, builder: GraphBuilder, llm, limit: int | None = None) -> None:
    """Build knowledge graph directly from PDF files in a directory."""
    docs = load_papers_dir(papers_dir)
    if not docs:
        logger.warning(f"No PDF files found in {papers_dir}")
        return

    if limit:
        docs = docs[:limit]

    total = len(docs)
    logger.info(f"Processing {total} papers from {papers_dir}")

    import time
    for idx, doc in enumerate(docs, 1):
        if is_paper_in_graph(builder.client, doc.filename):
            logger.info(f"[{idx}/{total}] Skipping '{doc.filename}': already extracted in Neo4j graph")
            continue

        start_time = time.time()
        logger.info(f"[{idx}/{total}] Processing document: '{doc.filename}'")
        extraction = extract_graph_elements(
            llm=llm,
            paper_title=doc.title,
            paper_filename=doc.filename,
            text=doc.full_text[:4000],
            max_text_chars=4000,
        )
        builder.build_from_extraction(extraction, authors=doc.authors)
        elapsed = time.time() - start_time
        logger.success(f"[{idx}/{total}] ✅ '{doc.filename}' finished in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Build Knowledge Graph in Neo4j")
    parser.add_argument(
        "--papers-dir",
        type=Path,
        default=Path("data/papers"),
        help="Directory with PDF papers (default: data/papers/)",
    )
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Build using already ingested papers in PostgreSQL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers to process (e.g. --limit 5)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset and clear existing Neo4j graph before building",
    )
    args = parser.parse_args()

    if args.reset:
        clear_graph()

    builder = GraphBuilder()
    llm = get_llm_client()

    if args.from_db:
        build_from_database(builder, llm, limit=args.limit)
    else:
        if args.papers_dir.exists() and any(args.papers_dir.glob("*.pdf")):
            build_from_directory(args.papers_dir, builder, llm, limit=args.limit)
        else:
            logger.info("No PDFs in directory; attempting to build from PostgreSQL...")
            build_from_database(builder, llm, limit=args.limit)

    stats = builder.client.run("MATCH (n) RETURN count(n) AS total_nodes")
    total_nodes = stats[0]["total_nodes"] if stats else 0
    logger.success(f"Graph construction completed! Total nodes in Neo4j: {total_nodes}")


if __name__ == "__main__":
    main()
