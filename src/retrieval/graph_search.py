"""
Graph search — parameterized Cypher queries to explore and query the Neo4j Knowledge Graph.
"""

from typing import Any
from loguru import logger

from src.db.neo4j_client import Neo4jClient, get_neo4j_client


class GraphSearch:
    def __init__(self, client: Neo4jClient | None = None):
        self.client = client or get_neo4j_client()

    def find_papers_by_author(self, author_name: str) -> list[dict[str, Any]]:
        """Find all papers written by a given author (case-insensitive substring match)."""
        cql = """
        MATCH (a:Author)-[:WROTE]->(p:Paper)
        WHERE toLower(a.name) CONTAINS toLower($author_name)
        RETURN a.name AS author, p.title AS title, p.filename AS filename
        """
        return self.client.run(cql, author_name=author_name)

    def find_concepts_by_paper(self, filename: str) -> list[dict[str, Any]]:
        """Find concepts, theorems, and methods associated with a paper."""
        cql = """
        MATCH (p:Paper {filename: $filename})-[r]->(e)
        RETURN type(r) AS relation, labels(e)[0] AS entity_type, e.name AS name, e.description AS description
        """
        return self.client.run(cql, filename=filename)

    def find_papers_by_concept(self, concept_name: str) -> list[dict[str, Any]]:
        """Find papers that discuss, prove, or use a given concept/theorem."""
        cql = """
        MATCH (p:Paper)-[r]->(e)
        WHERE toLower(e.name) CONTAINS toLower($concept_name)
        RETURN p.title AS title, p.filename AS filename, type(r) AS relation, e.name AS entity_name
        """
        return self.client.run(cql, concept_name=concept_name)

    def find_path_between_entities(
        self,
        source_name: str,
        target_name: str,
        max_depth: int = 3,
    ) -> list[dict[str, Any]]:
        """Find shortest paths between two entities (concepts, authors, papers, etc.)."""
        cql = f"""
        MATCH (s), (t)
        WHERE toLower(coalesce(s.name, s.title)) = toLower($source_name)
          AND toLower(coalesce(t.name, t.title)) = toLower($target_name)
        MATCH path = shortestPath((s)-[*..{max_depth}]-(t))
        RETURN [n in nodes(path) | coalesce(n.name, n.title)] AS nodes,
               [r in relationships(path) | type(r)] AS relationships,
               length(path) AS length
        LIMIT 5
        """
        return self.client.run(cql, source_name=source_name, target_name=target_name)

    def find_related_entities(
        self,
        entity_name: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Find all 1-hop connected neighbors of an entity."""
        cql = """
        MATCH (s)-[r]-(t)
        WHERE toLower(coalesce(s.name, s.title)) CONTAINS toLower($entity_name)
        RETURN coalesce(s.name, s.title) AS source,
               labels(s)[0] AS source_type,
               type(r) AS relation,
               coalesce(t.name, t.title) AS target,
               labels(t)[0] AS target_type
        LIMIT $limit
        """
        return self.client.run(cql, entity_name=entity_name, limit=limit)

    def get_graph_stats(self) -> dict[str, Any]:
        """Return summary statistics of the knowledge graph."""
        cql_nodes = """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(*) AS count
        """
        cql_rels = """
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(*) AS count
        """
        node_counts = {r["label"]: r["count"] for r in self.client.run(cql_nodes) if r.get("label")}
        rel_counts = {r["type"]: r["count"] for r in self.client.run(cql_rels) if r.get("type")}
        return {
            "total_nodes": sum(node_counts.values()),
            "total_relationships": sum(rel_counts.values()),
            "node_counts": node_counts,
            "relationship_counts": rel_counts,
        }

    def format_graph_context_for_prompt(self, related_items: list[dict[str, Any]]) -> str:
        """Format graph relationships into human-readable sentences for LLM prompt context."""
        if not related_items:
            return "No graph relations found for this query."
        lines = []
        for item in related_items:
            src = item.get("source", "")
            rel = item.get("relation", "").replace("_", " ").lower()
            tgt = item.get("target", "")
            lines.append(f"- ({item.get('source_type', '')}) {src} -> [{rel}] -> ({item.get('target_type', '')}) {tgt}")
        return "\n".join(lines)


# Singleton
_graph_search: GraphSearch | None = None


def get_graph_search() -> GraphSearch:
    global _graph_search
    if _graph_search is None:
        _graph_search = GraphSearch()
    return _graph_search
