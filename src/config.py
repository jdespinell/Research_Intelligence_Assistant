"""
Centralized configuration using pydantic-settings.
All settings are read from environment variables or .env file.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # PostgreSQL
    # -------------------------------------------------------------------------
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "ria"
    postgres_password: str = "ria_password"
    postgres_db: str = "ria_db"

    @computed_field  # type: ignore[misc]
    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # -------------------------------------------------------------------------
    # Neo4j
    # -------------------------------------------------------------------------
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "ria_neo4j_pass"

    # -------------------------------------------------------------------------
    # LLM Provider
    # -------------------------------------------------------------------------
    llm_provider: Literal["ollama", "gemini"] = "ollama"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # Gemini
    gemini_api_key: str = Field(default="", repr=False)
    gemini_model: str = "gemini-2.0-flash"

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dimension: int = 768

    # -------------------------------------------------------------------------
    # RAG
    # -------------------------------------------------------------------------
    chunk_size: int = 600
    chunk_overlap: int = 100
    retrieval_top_k: int = 5


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings instance (singleton)."""
    return Settings()
