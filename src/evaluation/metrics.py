"""
Evaluation metrics for RAG retrieval and response generation.

Retrieval metrics:
  - Precision@k
  - Recall@k
  - MRR (Mean Reciprocal Rank)

Generation metrics (LLM-as-a-judge):
  - Faithfulness (checks whether claims in the answer are supported by context)
  - Answer Relevance (checks whether the answer directly addresses the query)
"""

import json
import re
from typing import Any
from loguru import logger

from src.llm.base import LLMClient, Message


# ---------------------------------------------------------------------------
# Retrieval Metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: list[Any], relevant_ids: set[Any], k: int) -> float:
    """
    Precision@k = |{retrieved in top-k} ∩ {relevant}| / k
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant_ids)
    return hits / k


def recall_at_k(retrieved_ids: list[Any], relevant_ids: set[Any], k: int) -> float:
    """
    Recall@k = |{retrieved in top-k} ∩ {relevant}| / |relevant|
    """
    if not relevant_ids:
        return 1.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list[Any], relevant_ids: set[Any]) -> float:
    """
    Mean Reciprocal Rank for a single query:
    1 / rank of the first relevant item retrieved, or 0.0 if none.
    """
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Generation Metrics (LLM-as-Judge)
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """You are an impartial academic evaluator assessing whether an AI-generated answer is faithful to the provided context passages.

Context:
\"\"\"
{context}
\"\"\"

Question:
\"\"\"
{question}
\"\"\"

Answer:
\"\"\"
{answer}
\"\"\"

Task:
1. Break down the answer into individual factual statements/claims.
2. For each claim, check if it can be directly inferred from the Context passages.
3. Compute a score between 0.0 and 1.0 representing the proportion of claims supported by the context.
   (1.0 = completely faithful, 0.0 = completely hallucinated or unsupported).

Return ONLY a valid JSON object in this exact format:
{{
    "claims": ["claim 1", "claim 2"],
    "supported_claims": ["claim 1"],
    "score": 0.85,
    "reasoning": "Brief explanation"
}}
"""

RELEVANCE_PROMPT = """You are an impartial academic evaluator assessing whether an answer directly and effectively addresses the question asked.

Question:
\"\"\"
{question}
\"\"\"

Answer:
\"\"\"
{answer}
\"\"\"

Task:
Evaluate how well the answer answers the question.
- 1.0: Directly and comprehensively answers the question.
- 0.5: Partially answers the question or goes off-topic.
- 0.0: Completely irrelevant or refuses to answer without justification.

Return ONLY a valid JSON object in this exact format:
{{
    "score": 0.9,
    "reasoning": "Brief explanation"
}}
"""


def _extract_json(text: str) -> dict[str, Any]:
    """Helper to extract JSON from LLM response text."""
    # Match markdown json block if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Otherwise try direct bracket search
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return json.loads(text[start : end + 1])
    raise ValueError(f"Could not parse JSON from text: {text}")


def compute_faithfulness(
    llm: LLMClient,
    question: str,
    answer: str,
    context: str,
) -> dict[str, Any]:
    """
    Evaluate faithfulness using LLM-as-a-judge.
    """
    prompt = FAITHFULNESS_PROMPT.format(
        question=question,
        answer=answer,
        context=context,
    )
    try:
        response_text = llm.chat([Message(role="user", content=prompt)])
        result = _extract_json(response_text)
        score = float(result.get("score", 0.0))
        return {
            "score": max(0.0, min(1.0, score)),
            "reasoning": result.get("reasoning", ""),
            "claims": result.get("claims", []),
            "supported_claims": result.get("supported_claims", []),
        }
    except Exception as e:
        logger.warning(f"Error computing faithfulness: {e}")
        return {"score": 0.0, "reasoning": str(e), "claims": [], "supported_claims": []}


def compute_answer_relevance(
    llm: LLMClient,
    question: str,
    answer: str,
) -> dict[str, Any]:
    """
    Evaluate answer relevance using LLM-as-a-judge.
    """
    prompt = RELEVANCE_PROMPT.format(question=question, answer=answer)
    try:
        response_text = llm.chat([Message(role="user", content=prompt)])
        result = _extract_json(response_text)
        score = float(result.get("score", 0.0))
        return {
            "score": max(0.0, min(1.0, score)),
            "reasoning": result.get("reasoning", ""),
        }
    except Exception as e:
        logger.warning(f"Error computing relevance: {e}")
        return {"score": 0.0, "reasoning": str(e)}
