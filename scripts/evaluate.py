"""
scripts/evaluate.py

Evaluation CLI for the Research Intelligence Assistant.
Measures retrieval quality (keyword-based relevant chunks) and response quality
(faithfulness & answer relevance via LLM-as-a-judge).

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --dataset data/eval/eval_dataset.json --top-k 5
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
import numpy as np

from src.config import get_settings
from src.evaluation.metrics import (
    compute_answer_relevance,
    compute_faithfulness,
    mrr,
    precision_at_k,
    recall_at_k,
)
from src.llm.base import Message
from src.llm.factory import get_llm_client
from src.llm.prompts import RAG_SYSTEM, RAG_USER
from src.retrieval.vector_search import format_chunks_for_prompt, vector_search

settings = get_settings()


def run_evaluation(dataset_path: Path, top_k: int, output_dir: Path) -> dict:
    if not dataset_path.exists():
        logger.error(f"Evaluation dataset not found: {dataset_path}")
        sys.exit(1)

    items = json.loads(dataset_path.read_text(encoding="utf-8"))
    logger.info(f"Loaded {len(items)} evaluation queries from {dataset_path}")

    llm = get_llm_client()
    results = []

    for idx, item in enumerate(items, 1):
        q_id = item.get("id", f"q{idx}")
        question = item["question"]
        expected_answer = item.get("expected_answer", "")
        keywords = [k.lower() for k in item.get("relevant_keywords", [])]

        logger.info(f"[{idx}/{len(items)}] Evaluating query: '{question[:60]}...'")

        # 1. Retrieve chunks
        search_results = vector_search(question, top_k=top_k)

        # Keyword matching heuristic to determine relevant chunks
        retrieved_ids = [r.chunk_id for r in search_results]
        relevant_chunk_ids = set()
        for r in search_results:
            text_lower = r.content.lower()
            if any(kw in text_lower for kw in keywords):
                relevant_chunk_ids.add(r.chunk_id)

        # Retrieval metrics
        p_at_k = precision_at_k(retrieved_ids, relevant_chunk_ids, k=top_k)
        rec_at_k = recall_at_k(retrieved_ids, relevant_chunk_ids, k=top_k)
        mrr_score = mrr(retrieved_ids, relevant_chunk_ids)

        # 2. Generate answer
        chunks_text = format_chunks_for_prompt(search_results)
        user_prompt = RAG_USER.format(chunks=chunks_text, query=question)
        answer = llm.chat([
            Message(role="system", content=RAG_SYSTEM),
            Message(role="user", content=user_prompt),
        ])

        # 3. LLM-as-judge metrics
        faith_eval = compute_faithfulness(llm, question, answer, chunks_text)
        rel_eval = compute_answer_relevance(llm, question, answer)

        logger.info(
            f"   P@{top_k}={p_at_k:.2f} | Faithfulness={faith_eval['score']:.2f} | "
            f"Relevance={rel_eval['score']:.2f}"
        )

        results.append({
            "id": q_id,
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": answer,
            "retrieved_chunks_count": len(search_results),
            "relevant_chunks_found": len(relevant_chunk_ids),
            "precision_at_k": p_at_k,
            "recall_at_k": rec_at_k,
            "mrr": mrr_score,
            "faithfulness": faith_eval["score"],
            "faithfulness_reasoning": faith_eval["reasoning"],
            "relevance": rel_eval["score"],
            "relevance_reasoning": rel_eval["reasoning"],
        })

    # Summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "model": settings.gemini_model if settings.llm_provider == "gemini" else settings.ollama_model,
        "embedding_model": settings.embedding_model,
        "top_k": top_k,
        "num_queries": len(results),
        "mean_precision_at_k": float(np.mean([r["precision_at_k"] for r in results])),
        "mean_recall_at_k": float(np.mean([r["recall_at_k"] for r in results])),
        "mean_mrr": float(np.mean([r["mrr"] for r in results])),
        "mean_faithfulness": float(np.mean([r["faithfulness"] for r in results])),
        "mean_relevance": float(np.mean([r["relevance"] for r in results])),
        "detailed_results": results,
    }

    # Save reports
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation_report.json"
    md_path = output_dir / "evaluation_report.md"

    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Generate Markdown Report
    md_content = f"""# 📊 RAG Evaluation Report

- **Date:** {summary['timestamp']}
- **LLM Model:** `{summary['model']}`
- **Embedding Model:** `{summary['embedding_model']}`
- **Top-k:** {summary['top_k']}
- **Evaluated Queries:** {summary['num_queries']}

## 📈 Overall Metrics Summary

| Metric | Mean Score | Target Threshold |
| :--- | :--- | :--- |
| **Precision@{top_k}** | {summary['mean_precision_at_k']:.3f} | ≥ 0.60 |
| **Recall@{top_k}** | {summary['mean_recall_at_k']:.3f} | ≥ 0.50 |
| **MRR** | {summary['mean_mrr']:.3f} | ≥ 0.70 |
| **Faithfulness (No Hallucination)** | {summary['mean_faithfulness']:.3f} | ≥ 0.80 |
| **Answer Relevance** | {summary['mean_relevance']:.3f} | ≥ 0.80 |

## 📝 Query Breakdown

| ID | Question | P@{top_k} | Faithfulness | Relevance |
| :--- | :--- | :--- | :--- | :--- |
"""
    for r in results:
        md_content += f"| {r['id']} | {r['question'][:50]}... | {r['precision_at_k']:.2f} | {r['faithfulness']:.2f} | {r['relevance']:.2f} |\n"

    md_path.write_text(md_content, encoding="utf-8")
    logger.success(f"Reports saved to:\n  - {json_path}\n  - {md_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval and answers")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/eval/eval_dataset.json"),
        help="Path to evaluation questions JSON",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved chunks (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/eval"),
        help="Directory to save evaluation reports",
    )
    args = parser.parse_args()

    run_evaluation(args.dataset, top_k=args.top_k, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
