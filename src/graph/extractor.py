"""
Entity and relationship extraction from paper text using LLM.

Extracts:
  - Entities: Author, Concept, Theorem, Method
  - Relations: WROTE, DISCUSSES, PROVES, USES_METHOD, RELATED_TO, GENERALIZES, CITES
"""

import json
import re
from typing import Any, Literal
from loguru import logger
from pydantic import BaseModel, Field

from src.evaluation.metrics import _extract_json
from src.llm.base import LLMClient, Message

EntityType = Literal["Author", "Concept", "Theorem", "Method"]
RelationType = Literal[
    "WROTE",
    "DISCUSSES",
    "PROVES",
    "USES_METHOD",
    "RELATED_TO",
    "GENERALIZES",
    "CITES",
]


class ExtractedEntity(BaseModel):
    name: str = Field(description="Normalized name of the entity")
    type: EntityType = Field(description="Entity category")
    description: str | None = Field(default=None, description="Short definition or role if mentioned")


class ExtractedRelation(BaseModel):
    source: str = Field(description="Source entity name")
    source_type: str = Field(description="Source entity type (e.g. Paper, Author, Concept, Theorem)")
    relation: RelationType = Field(description="Relationship predicate")
    target: str = Field(description="Target entity name")
    target_type: str = Field(description="Target entity type (e.g. Paper, Concept, Theorem, Method)")


class GraphExtractionResult(BaseModel):
    paper_title: str
    paper_filename: str
    entities: list[ExtractedEntity] = []
    relations: list[ExtractedRelation] = []


EXTRACTION_SYSTEM_PROMPT = """You are an expert research ontology engineer in mathematics and computer science.
Your job is to read scientific paper text and extract key semantic entities and directed relationships
to build a Knowledge Graph in Neo4j.

Allowed Entity Types:
- Author: Researcher names (e.g., 'A. G. Kouchnirenko', 'Mutsuo Oka', 'John Milnor')
- Concept: Mathematical definitions, objects, or geometric notions (e.g., 'Newton polyhedron', 'Khovanskii non-degeneracy', 'Milnor fibration', 'topological triviality')
- Theorem: Named theorems or key results (e.g., "Oka's theorem", "Kouchnirenko's formula", "Leksin's theorem")
- Method: Mathematical or algorithmic techniques (e.g., 'resolution of singularities', 'toric compactification', 'vector field integration')

Allowed Relationships:
- (Author)-[:WROTE]->(Paper)
- (Paper)-[:DISCUSSES]->(Concept)
- (Paper)-[:PROVES]->(Theorem)
- (Paper)-[:USES_METHOD]->(Method)
- (Concept)-[:RELATED_TO]->(Concept)
- (Theorem)-[:GENERALIZES]->(Theorem)
- (Paper)-[:CITES]->(Paper)

Rules:
1. Normalize names (e.g. use standard capitalization, avoid abbreviations like 'Thm.' -> 'Theorem').
2. Keep entity names concise (2-5 words).
3. Do NOT invent information not present in the text.
4. Output strictly valid JSON matching the specified schema.
"""

EXTRACTION_USER_TEMPLATE = """Paper Title: {title}
Paper Filename: {filename}

Text to analyze:
\"\"\"
{text}
\"\"\"

Extract the knowledge graph entities and relationships. Return ONLY a valid JSON object matching this schema:
{{
  "entities": [
    {{"name": "...", "type": "Concept|Theorem|Method|Author", "description": "..."}}
  ],
  "relations": [
    {{"source": "...", "source_type": "...", "relation": "DISCUSSES|PROVES|USES_METHOD|RELATED_TO|GENERALIZES|WROTE|CITES", "target": "...", "target_type": "..."}}
  ]
}}
"""


def normalize_entity_name(name: str) -> str:
    """Clean and normalize entity names for consistent graph nodes."""
    clean = name.strip()
    clean = re.sub(r"^(the|a|an)\s+", "", clean, flags=re.IGNORECASE)
    clean = clean.strip(".,;:\"'()")
    return clean


def extract_graph_elements(
    llm: LLMClient,
    paper_title: str,
    paper_filename: str,
    text: str,
    max_text_chars: int = 6000,
) -> GraphExtractionResult:
    """
    Extract entities and relationships from paper text using LLM.
    """
    # Sample beginning and representative content if text is too long
    sampled_text = text[:max_text_chars] if len(text) > max_text_chars else text

    user_prompt = EXTRACTION_USER_TEMPLATE.format(
        title=paper_title,
        filename=paper_filename,
        text=sampled_text,
    )

    try:
        response_text = llm.chat([
            Message(role="system", content=EXTRACTION_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ])
        raw_json = _extract_json(response_text)

        entities: list[ExtractedEntity] = []
        for e in raw_json.get("entities", []):
            name = normalize_entity_name(e.get("name", ""))
            etype = e.get("type", "Concept")
            if name and etype in ["Author", "Concept", "Theorem", "Method"]:
                entities.append(
                    ExtractedEntity(
                        name=name,
                        type=etype,
                        description=e.get("description"),
                    )
                )

        relations: list[ExtractedRelation] = []
        for r in raw_json.get("relations", []):
            src = normalize_entity_name(r.get("source", ""))
            tgt = normalize_entity_name(r.get("target", ""))
            rel = r.get("relation", "")
            if src and tgt and rel in [
                "WROTE",
                "DISCUSSES",
                "PROVES",
                "USES_METHOD",
                "RELATED_TO",
                "GENERALIZES",
                "CITES",
            ]:
                relations.append(
                    ExtractedRelation(
                        source=src,
                        source_type=r.get("source_type", "Concept"),
                        relation=rel,
                        target=tgt,
                        target_type=r.get("target_type", "Concept"),
                    )
                )

        logger.info(
            f"Extracted {len(entities)} entities, {len(relations)} relations from '{paper_filename}'"
        )
        return GraphExtractionResult(
            paper_title=paper_title,
            paper_filename=paper_filename,
            entities=entities,
            relations=relations,
        )

    except Exception as e:
        logger.error(f"Failed to extract graph from '{paper_filename}': {e}")
        return GraphExtractionResult(
            paper_title=paper_title,
            paper_filename=paper_filename,
            entities=[],
            relations=[],
        )
