"""
Neo4j client — connection helper and basic Cypher utilities.
"""

from contextlib import contextmanager
from typing import Any, Generator

from loguru import logger
from neo4j import GraphDatabase, Session

from src.config import get_settings

settings = get_settings()


class Neo4jClient:
    """Thin wrapper around the Neo4j driver."""

    def __init__(self):
        self._driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        logger.info(f"Neo4jClient initialized | uri={settings.neo4j_uri}")

    def verify(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Neo4j connectivity error: {e}")
            return False

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        with self._driver.session() as s:
            yield s

    def run(self, query: str, **params) -> list[dict[str, Any]]:
        with self.session() as s:
            result = s.run(query, **params)
            return [dict(r) for r in result]

    def init_constraints(self) -> None:
        """Create uniqueness constraints for graph nodes."""
        constraints = [
            "CREATE CONSTRAINT paper_filename IF NOT EXISTS FOR (p:Paper) REQUIRE p.filename IS UNIQUE",
            "CREATE CONSTRAINT author_name IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT concept_name IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT theorem_name IF NOT EXISTS FOR (t:Theorem) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT method_name IF NOT EXISTS FOR (m:Method) REQUIRE m.name IS UNIQUE",
        ]
        for cql in constraints:
            self.run(cql)
        logger.success("Neo4j constraints initialized ✅")

    def close(self) -> None:
        self._driver.close()


# Singleton
_neo4j_client: Neo4jClient | None = None


def get_neo4j_client() -> Neo4jClient:
    global _neo4j_client
    if _neo4j_client is None:
        _neo4j_client = Neo4jClient()
    return _neo4j_client
