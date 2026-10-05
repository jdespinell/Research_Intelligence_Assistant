"""
Prompt templates for all RAG and extraction tasks.
"""

# ---------------------------------------------------------------------------
# RAG — Question answering with citations
# ---------------------------------------------------------------------------

RAG_SYSTEM = """\
You are a scientific research assistant specialized in mathematics.
Your task is to answer questions based ONLY on the provided document fragments.
Rules:
- Cite the exact source (paper title and page number) for every claim.
- If the fragments do not contain enough information, say so explicitly.
- Do NOT fabricate or infer information not present in the fragments.
- Respond in the same language as the question.
"""

RAG_USER = """\
Document fragments:
{chunks}

Question: {query}
"""

# ---------------------------------------------------------------------------
# Entity & relation extraction (Fase 3 — Knowledge Graph)
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """\
You are an expert in mathematical literature analysis.
Extract named entities and relations from the provided scientific text.
Return ONLY valid JSON. No explanation, no markdown, only the JSON object.
"""

EXTRACTION_USER = """\
Extract entities and relations from the following text.

Text:
{text}

Return a JSON object with this exact structure:
{{
  "entities": [
    {{"type": "Concept|Theorem|Method|Author|Paper", "name": "...", "description": "..."}}
  ],
  "relations": [
    {{"source": "entity_name", "type": "DISCUSSES|PROVES|USES|CITES|GENERALIZES|RELATED_TO", "target": "entity_name"}}
  ]
}}
"""

# ---------------------------------------------------------------------------
# LLM-as-judge — Faithfulness evaluation (Fase 2)
# ---------------------------------------------------------------------------

FAITHFULNESS_SYSTEM = """\
You are evaluating whether an AI-generated answer is faithful to provided source fragments.
Faithful means every claim in the answer is directly supported by the fragments.
Return ONLY a JSON object with no additional text.
"""

FAITHFULNESS_USER = """\
Source fragments:
{chunks}

Generated answer:
{answer}

For each claim in the answer, determine if it is supported by the fragments.
Return:
{{
  "score": <float between 0.0 and 1.0>,
  "unsupported_claims": ["claim 1", "claim 2", ...]
}}
"""

# ---------------------------------------------------------------------------
# Hybrid GraphRAG — Textual evidence + Knowledge Graph relations (Fase 4)
# ---------------------------------------------------------------------------

HYBRID_RAG_SYSTEM = """\
You are an expert scientific research intelligence assistant in mathematics.
Your task is to synthesize an answer to the question using BOTH:
1. Textual evidence (exact passages from scientific papers with page numbers)
2. Ontological evidence from the Knowledge Graph (relationships between concepts, theorems, authors, and papers).

Rules:
- Synthesize relationships clearly (e.g. "Paper X proves Theorem Y, which generalizes Result Z").
- Cite exact paper sources and pages for textual claims.
- Explicitly reference relations from the Knowledge Graph when explaining connections between authors or concepts.
- If information is insufficient to answer completely, acknowledge the limitations.
- Respond in the same language as the question.
"""

HYBRID_RAG_USER = """\
Textual Evidence (Retrieved Chunks):
{chunks}

Knowledge Graph Relations:
{graph_context}

Question: {query}
"""
