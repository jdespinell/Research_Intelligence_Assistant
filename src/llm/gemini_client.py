"""
Gemini API LLM client implementation.
Uses the official google-genai SDK.
"""

from google import genai
from google.genai import types
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.llm.base import LLMClient, Message


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required when LLM_PROVIDER='gemini'. "
                "Add it to your .env file."
            )
        self.model = model
        self.client = genai.Client(api_key=api_key)
        logger.info(f"GeminiClient initialized | model={model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def chat(self, messages: list[Message], **kwargs) -> str:
        # Separate system message from conversation history
        system_instruction = None
        conversation: list[types.Content] = []

        for m in messages:
            if m.role == "system":
                system_instruction = m.content
            else:
                role = "user" if m.role == "user" else "model"
                conversation.append(
                    types.Content(
                        role=role,
                        parts=[types.Part(text=m.content)],
                    )
                )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=kwargs.get("temperature", 0.2),
            max_output_tokens=kwargs.get("max_tokens", 4096),
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=conversation,
            config=config,
        )
        return response.text
