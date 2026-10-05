"""
LLM client factory.
Instantiates the correct LLMClient based on LLM_PROVIDER in config.
"""

from functools import lru_cache

from loguru import logger

from src.config import get_settings
from src.llm.base import LLMClient


@lru_cache
def get_llm_client() -> LLMClient:
    """
    Return a cached LLMClient instance.
    Provider is selected via the LLM_PROVIDER environment variable.

    Supported values:
      - "ollama"  → OllamaClient (local inference)
      - "gemini"  → GeminiClient (Google API)
    """
    settings = get_settings()
    provider = settings.llm_provider
    logger.info(f"LLM provider: {provider}")

    if provider == "ollama":
        from src.llm.ollama_client import OllamaClient
        return OllamaClient(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
        )

    if provider == "gemini":
        from src.llm.gemini_client import GeminiClient
        return GeminiClient(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER='{provider}'. Valid options: 'ollama', 'gemini'."
    )
