"""
Abstract base class for LLM clients.
All providers (Ollama, Gemini, ...) must implement this interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Message:
    role: str   # "user" | "assistant" | "system"
    content: str


class LLMClient(ABC):
    """Minimal interface every LLM backend must implement."""

    @abstractmethod
    def chat(self, messages: list[Message], **kwargs) -> str:
        """
        Send a list of messages and return the model's text response.

        Args:
            messages: Conversation history as a list of Message objects.
            **kwargs: Provider-specific overrides (temperature, max_tokens, etc.)

        Returns:
            The model's response as a plain string.
        """
        ...

    def complete(self, prompt: str, **kwargs) -> str:
        """
        Convenience method: wrap a single user prompt and call chat().
        """
        return self.chat([Message(role="user", content=prompt)], **kwargs)
