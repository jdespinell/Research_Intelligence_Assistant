"""
API routes — query endpoint and corpus stats.
"""

import json
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel

from src.llm.factory import get_llm_client
from src.llm.prompts import RAG_SYSTEM, RAG_USER
from src.llm.base import Message
from src.retrieval.vector_search import SearchResult, format_chunks_for_prompt, vector_search

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    paper_filter: list[str] | None = None


class ChunkSource(BaseModel):
    chunk_id: int
    paper_title: str
    paper_filename: str
    page_number: int
    score: float
    content: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[ChunkSource]
    model: str


class CorpusStats(BaseModel):
    num_papers: int
    num_chunks: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Answer a question about the ingested papers using RAG.
    Returns the answer and the source chunks used to generate it.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(f"Query: '{request.question[:80]}'")

    # 1. Retrieve relevant chunks
    results: list[SearchResult] = vector_search(
        query=request.question,
        top_k=request.top_k,
        paper_filter=request.paper_filter,
    )

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No relevant chunks found. Make sure papers have been ingested.",
        )

    # 2. Build prompt context
    chunks_text = format_chunks_for_prompt(results)
    user_prompt = RAG_USER.format(chunks=chunks_text, query=request.question)

    # 3. Call LLM
    llm = get_llm_client()
    answer = llm.chat([
        Message(role="system", content=RAG_SYSTEM),
        Message(role="user", content=user_prompt),
    ])

    # 4. Build response
    sources = [
        ChunkSource(
            chunk_id=r.chunk_id,
            paper_title=r.paper_title,
            paper_filename=r.paper_filename,
            page_number=r.page_number,
            score=round(r.score, 4),
            content=r.content[:300] + "..." if len(r.content) > 300 else r.content,
        )
        for r in results
    ]

    from src.config import get_settings
    settings = get_settings()
    model_name = (
        settings.ollama_model
        if settings.llm_provider == "ollama"
        else settings.gemini_model
    )

    return QueryResponse(answer=answer, sources=sources, model=model_name)


@router.get("/stats", response_model=CorpusStats)
def corpus_stats():
    """Return basic statistics about the ingested corpus."""
    from src.db.postgres import get_connection
    try:
        with get_connection() as conn:
            num_papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            num_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return CorpusStats(num_papers=num_papers, num_chunks=num_chunks)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {e}")


@router.get("/papers")
def list_papers():
    """List all ingested papers."""
    from src.db.postgres import get_connection
    try:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT filename, title, authors, num_pages, ingested_at FROM papers ORDER BY ingested_at DESC"
            ).fetchall()
        return [
            {
                "filename": r[0],
                "title": r[1],
                "authors": r[2] or [],
                "num_pages": r[3],
                "ingested_at": r[4].isoformat() if r[4] else None,
            }
            for r in rows
        ]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {e}")


# ---------------------------------------------------------------------------
# Knowledge Graph Endpoints
# ---------------------------------------------------------------------------

@router.get("/graph/stats")
def graph_stats():
    """Return Knowledge Graph statistics from Neo4j."""
    from src.retrieval.graph_search import get_graph_search
    try:
        gs = get_graph_search()
        return gs.get_graph_stats()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Neo4j error: {e}")


@router.get("/graph/explore")
def graph_explore(entity: str = Query(..., description="Entity name to explore"), limit: int = 20):
    """Find immediate neighbors and relationships of an entity in the graph."""
    from src.retrieval.graph_search import get_graph_search
    try:
        gs = get_graph_search()
        results = gs.find_related_entities(entity_name=entity, limit=limit)
        return {"entity": entity, "relations": results}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Neo4j error: {e}")


@router.get("/graph/path")
def graph_path(
    source: str = Query(..., description="Source entity name"),
    target: str = Query(..., description="Target entity name"),
    max_depth: int = 3,
):
    """Find multi-hop paths between two entities in the knowledge graph."""
    from src.retrieval.graph_search import get_graph_search
    try:
        gs = get_graph_search()
        paths = gs.find_path_between_entities(source_name=source, target_name=target, max_depth=max_depth)
        return {"source": source, "target": target, "paths": paths}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Neo4j error: {e}")
