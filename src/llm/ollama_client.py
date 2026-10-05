"""
Ollama LLM client implementation.
Requires Ollama running locally (default: http://localhost:11434).
"""

import ollama
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.llm.base import LLMClient, Message


class OllamaClient(LLMClient):
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.client = ollama.Client(host=base_url)
        logger.info(f"OllamaClient initialized | model={model} | url={base_url}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def chat(self, messages: list[Message], **kwargs) -> str:
        ollama_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        response = self.client.chat(
            model=self.model,
            messages=ollama_messages,
            options=kwargs or None,
        )
        return response.message.content

    def is_available(self) -> bool:
        """Check if Ollama is reachable and the model is available."""
        try:
            models = self.client.list()
            names = [m.model for m in models.models]
            return any(self.model in name for name in names)
        except Exception as e:
            logger.warning(f"Ollama not reachable: {e}")
            return False
