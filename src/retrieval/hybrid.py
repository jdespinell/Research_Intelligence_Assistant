"""
Hybrid GraphRAG retrieval — Combines dense semantic vector search (pgvector)
with ontological graph traversal (Neo4j) guided by an intelligent query router.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Literal
from loguru import logger

from src.llm.prompts import HYBRID_RAG_SYSTEM, HYBRID_RAG_USER, RAG_SYSTEM, RAG_USER
from src.retrieval.graph_search import GraphSearch, get_graph_search
from src.retrieval.vector_search import SearchResult, format_chunks_for_prompt, vector_search

QueryType = Literal["factual", "relational", "hybrid"]


@dataclass
class HybridResult:
    query: str
    query_type: QueryType
    mode_used: str
    chunks: list[SearchResult] = field(default_factory=list)
    graph_relations: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    user_prompt: str = ""


# ---------------------------------------------------------------------------
# Query Router
# ---------------------------------------------------------------------------

RELATIONAL_PATTERNS = [
    r"\brelaci[oó]n\b",
    r"\brelationships?\b",
    r"\bconect(an?|ar)\b",
    r"\bautores?\b",
    r"\bcamino\b",
    r"\bentre\s+.*\s+y\b",
    r"\bbetween\s+.*\s+and\b",
    r"\bwho\s+wrote\b",
    r"\bqui[eé]n\s+escribi[oó]\b",
    r"\bgeneraliz(a|ation|es)\b",
    r"\bcompare\b",
    r"\bconnections?\b",
    r"\bgraph\b",
    r"\bnetwork\b",
]

FACTUAL_PATTERNS = [
    r"\bqu[eé]\s+es\b",
    r"\bwhat\s+is\b",
    r"\bdefini[cç][ií][oó]n\b",
    r"\bdefine\b",
    r"\benunciad[oó]\b",
    r"\bstatement\b",
    r"\bf[oó]rmula\b",
    r"\bformula\b",
    r"\bproof\s+of\b",
    r"\bdemostraci[oó]n\s+de\b",
]


def classify_query(query: str) -> QueryType:
    """
    Classify user query into:
      - 'relational': Multi-hop, author connections, concept paths -> Graph prioritized
      - 'factual': Direct definition, theorem statement, specific claim -> Vector prioritized
      - 'hybrid': High-complexity questions needing both evidence & ontological relations
    """
    q = query.lower()

    has_relational = any(re.search(pat, q, re.IGNORECASE) for pat in RELATIONAL_PATTERNS)
    has_factual = any(re.search(pat, q, re.IGNORECASE) for pat in FACTUAL_PATTERNS)

    if has_relational and not has_factual:
        return "relational"
    if has_factual and not has_relational:
        return "factual"
    return "hybrid"


def extract_potential_entities(query: str) -> list[str]:
    """
    Extract entity candidates from a natural language question
    (e.g., quoted phrases, capitalized multi-word phrases, or domain terms).
    """
    candidates = []

    # 1. Quoted terms: "Khovanskii non-degeneracy"
    quotes = re.findall(r"[\"']([^\"']+)[\"']", query)
    candidates.extend(quotes)

    # 2. Capitalized phrases: e.g. "Milnor Fibration", "Newton Polyhedron"
    cap_phrases = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", query)
    candidates.extend(cap_phrases)

    # 3. Known key domain terms
    domain_terms = [
        "Khovanskii non-degeneracy",
        "Newton polyhedron",
        "Newton non-degeneracy",
        "Milnor fibration",
        "Milnor number",
        "topological triviality",
        "isolated singularity",
        "Kouchnirenko",
        "Oka",
        "Milnor",
        "Khovanskii",
        "Arnold",
    ]
    q_lower = query.lower()
    for term in domain_terms:
        if term.lower() in q_lower and term not in candidates:
            candidates.append(term)

    # Deduplicate preserving order
    seen = set()
    result = []
    for c in candidates:
        norm = c.strip().lower()
        if norm and norm not in seen:
            seen.add(norm)
            result.append(c.strip())

    return result


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    def __init__(self, graph_search: GraphSearch | None = None):
        self.graph_search = graph_search or get_graph_search()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["auto", "vector", "graph", "hybrid"] = "auto",
        paper_filter: list[str] | None = None,
    ) -> HybridResult:
        """
        Execute hybrid retrieval based on query classification and specified mode.
        """
        detected_type = classify_query(query)
        effective_mode = mode

        if mode == "auto":
            if detected_type == "factual":
                effective_mode = "hybrid"  # Hybrid provides both context & graph relations if available
            elif detected_type == "relational":
                effective_mode = "hybrid"
            else:
                effective_mode = "hybrid"

        logger.info(f"Hybrid retrieval: mode={effective_mode} (type={detected_type}) | '{query[:60]}...'")

        chunks: list[SearchResult] = []
        graph_relations: list[dict[str, Any]] = []

        # 1. Vector Search
        if effective_mode in ["auto", "vector", "hybrid"]:
            try:
                chunks = vector_search(query=query, top_k=top_k, paper_filter=paper_filter)
            except Exception as e:
                logger.error(f"Vector search failed during hybrid retrieve: {e}")

        # 2. Graph Search
        if effective_mode in ["auto", "graph", "hybrid"]:
            entities = extract_potential_entities(query)

            # Also consider paper titles from retrieved chunks to look up connected concepts
            if chunks:
                for c in chunks[:3]:
                    if c.paper_title and c.paper_title not in entities:
                        entities.append(c.paper_title)

            # Look for paths between pairs of entities if multiple are detected
            if len(entities) >= 2:
                paths = self.graph_search.find_path_between_entities(entities[0], entities[1])
                for p in paths:
                    nodes = p.get("nodes", [])
                    rels = p.get("relationships", [])
                    for i in range(len(rels)):
                        graph_relations.append({
                            "source": nodes[i],
                            "source_type": "Entity",
                            "relation": rels[i],
                            "target": nodes[i + 1],
                            "target_type": "Entity",
                        })

            # Retrieve 1-hop neighbors for detected entities
            for ent in entities[:4]:
                rels = self.graph_search.find_related_entities(ent, limit=8)
                for r in rels:
                    if r not in graph_relations:
                        graph_relations.append(r)

        # 3. Format prompts
        chunks_text = format_chunks_for_prompt(chunks) if chunks else "No relevant passages found."
        graph_text = self.graph_search.format_graph_context_for_prompt(graph_relations)

        if effective_mode == "vector" or (not graph_relations and effective_mode == "auto"):
            system_prompt = RAG_SYSTEM
            user_prompt = RAG_USER.format(chunks=chunks_text, query=query)
        else:
            system_prompt = HYBRID_RAG_SYSTEM
            user_prompt = HYBRID_RAG_USER.format(
                chunks=chunks_text,
                graph_context=graph_text,
                query=query,
            )

        return HybridResult(
            query=query,
            query_type=detected_type,
            mode_used=effective_mode,
            chunks=chunks,
            graph_relations=graph_relations,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )


# Singleton
_hybrid_retriever: HybridRetriever | None = None


def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever
    if _hybrid_retriever is None:
        _hybrid_retriever = HybridRetriever()
    return _hybrid_retriever
