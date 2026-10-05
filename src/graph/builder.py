"""
Graph builder — persists extracted entities and relations into Neo4j.
Uses idempotent Cypher MERGE queries with uniqueness constraints.
"""

from typing import Any
from loguru import logger

from src.db.neo4j_client import Neo4jClient, get_neo4j_client
from src.graph.extractor import ExtractedEntity, ExtractedRelation, GraphExtractionResult


class GraphBuilder:
    def __init__(self, client: Neo4jClient | None = None):
        self.client = client or get_neo4j_client()
        self.client.init_constraints()

    def insert_paper(self, filename: str, title: str, authors: list[str] | None = None) -> None:
        """Insert a Paper node and connect Authors."""
        cql_paper = """
        MERGE (p:Paper {filename: $filename})
        ON CREATE SET p.title = $title, p.created_at = timestamp()
        ON MATCH SET p.title = $title
        """
        self.client.run(cql_paper, filename=filename, title=title)

        if authors:
            for author in authors:
                author_name = author.strip()
                if author_name:
                    cql_author = """
                    MERGE (a:Author {name: $author_name})
                    WITH a
                    MATCH (p:Paper {filename: $filename})
                    MERGE (a)-[:WROTE]->(p)
                    """
                    self.client.run(cql_author, author_name=author_name, filename=filename)

    def insert_entity(self, entity: ExtractedEntity) -> None:
        """Insert an entity node with its dynamic label."""
        label = entity.type  # Concept, Theorem, Method, Author
        cql = f"""
        MERGE (n:{label} {{name: $name}})
        ON CREATE SET n.description = $description, n.created_at = timestamp()
        ON MATCH SET n.description = coalesce($description, n.description)
        """
        self.client.run(cql, name=entity.name, description=entity.description)

    def insert_relation(
        self,
        paper_filename: str,
        rel: ExtractedRelation,
    ) -> None:
        """
        Insert a directed relationship between two entities.
        If source/target refers to the current paper or filename, links to Paper node.
        """
        source_name = rel.source
        target_name = rel.target

        # Handle paper self-references
        is_source_paper = rel.source_type == "Paper" or source_name in [paper_filename, "this paper", "the paper"]
        is_target_paper = rel.target_type == "Paper" or target_name in [paper_filename, "this paper", "the paper"]

        source_label = "Paper" if is_source_paper else rel.source_type
        target_label = "Paper" if is_target_paper else rel.target_type

        # Adjust query according to whether node is Paper (keyed on filename) or other (keyed on name)
        source_match = "MATCH (s:Paper {filename: $paper_filename})" if is_source_paper else f"MERGE (s:{source_label} {{name: $source_name}})"
        target_match = "MATCH (t:Paper {filename: $paper_filename})" if is_target_paper else f"MERGE (t:{target_label} {{name: $target_name}})"

        cql = f"""
        {source_match}
        WITH s
        {target_match}
        WITH s, t
        MERGE (s)-[r:{rel.relation}]->(t)
        ON CREATE SET r.created_at = timestamp()
        """
        try:
            self.client.run(
                cql,
                paper_filename=paper_filename,
                source_name=source_name,
                target_name=target_name,
            )
        except Exception as e:
            logger.warning(f"Could not insert relation {source_name} -[{rel.relation}]-> {target_name}: {e}")

    def build_from_extraction(
        self,
        extraction: GraphExtractionResult,
        authors: list[str] | None = None,
    ) -> dict[str, int]:
        """
        Build or update the graph for a single paper extraction result.
        Returns counts of nodes and relations inserted.
        """
        logger.info(f"Building graph for '{extraction.paper_filename}'...")

        # 1. Insert Paper & Authors
        self.insert_paper(
            filename=extraction.paper_filename,
            title=extraction.paper_title,
            authors=authors,
        )

        # 2. Insert extracted entities
        for entity in extraction.entities:
            self.insert_entity(entity)

        # 3. Insert relations
        for rel in extraction.relations:
            self.insert_relation(extraction.paper_filename, rel)

        logger.success(
            f"Graph updated for '{extraction.paper_filename}': "
            f"{len(extraction.entities)} entities, {len(extraction.relations)} relations"
        )
        return {
            "entities": len(extraction.entities),
            "relations": len(extraction.relations),
        }
