"""
scripts/check_connectivity.py
Verifies that PostgreSQL, Neo4j, and the configured LLM are reachable.
Run with: python scripts/check_connectivity.py
"""

import sys
from pathlib import Path

# Make sure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

import psycopg
from loguru import logger
from neo4j import GraphDatabase

from src.config import get_settings

settings = get_settings()


def check_postgres() -> bool:
    logger.info("Checking PostgreSQL...")
    try:
        conn_str = (
            f"host={settings.postgres_host} "
            f"port={settings.postgres_port} "
            f"dbname={settings.postgres_db} "
            f"user={settings.postgres_user} "
            f"password={settings.postgres_password}"
        )
        with psycopg.connect(conn_str) as conn:
            row = conn.execute("SELECT version()").fetchone()
            logger.success(f"PostgreSQL OK → {row[0][:50]}...")
            return True
    except Exception as e:
        logger.error(f"PostgreSQL FAILED: {e}")
        return False


def check_neo4j() -> bool:
    logger.info("Checking Neo4j...")
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        driver.verify_connectivity()
        with driver.session() as session:
            result = session.run("RETURN 'Neo4j OK' AS msg")
            msg = result.single()["msg"]
            logger.success(f"Neo4j OK → {msg}")
        driver.close()
        return True
    except Exception as e:
        logger.error(f"Neo4j FAILED: {e}")
        return False


def check_llm() -> bool:
    provider = settings.llm_provider
    logger.info(f"Checking LLM provider: {provider}...")
    try:
        if provider == "ollama":
            from src.llm.ollama_client import OllamaClient
            client = OllamaClient(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
            )
            available = client.is_available()
            if available:
                logger.success(f"Ollama OK → model '{settings.ollama_model}' found")
            else:
                logger.warning(
                    f"Ollama reachable but model '{settings.ollama_model}' not found. "
                    f"Run: ollama pull {settings.ollama_model}"
                )
            return available

        elif provider == "gemini":
            from src.llm.gemini_client import GeminiClient
            client = GeminiClient(
                api_key=settings.gemini_api_key,
                model=settings.gemini_model,
            )
            response = client.complete("Reply with exactly: OK")
            logger.success(f"Gemini OK → response: {response.strip()[:30]}")
            return True

    except Exception as e:
        logger.error(f"LLM ({provider}) FAILED: {e}")
        return False


def main():
    logger.info("=" * 50)
    logger.info("Research Intelligence Assistant — Connectivity Check")
    logger.info("=" * 50)

    results = {
        "PostgreSQL": check_postgres(),
        "Neo4j": check_neo4j(),
        f"LLM ({settings.llm_provider})": check_llm(),
    }

    logger.info("=" * 50)
    logger.info("Summary:")
    all_ok = True
    for service, ok in results.items():
        status = "✅ OK" if ok else "❌ FAILED"
        logger.info(f"  {service}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        logger.success("All services are reachable. Ready to go!")
        sys.exit(0)
    else:
        logger.error("Some services failed. Check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
