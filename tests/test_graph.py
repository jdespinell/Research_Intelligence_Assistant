"""
Tests for Knowledge Graph extractor, builder, and graph search.
"""

from src.db.neo4j_client import get_neo4j_client
from src.graph.builder import GraphBuilder
from src.graph.extractor import ExtractedEntity, ExtractedRelation, GraphExtractionResult, normalize_entity_name
from src.retrieval.graph_search import GraphSearch


def test_normalize_entity_name():
    assert normalize_entity_name("  The Newton Polyhedron. ") == "Newton Polyhedron"
    assert normalize_entity_name("a Milnor fibration,") == "Milnor fibration"
    assert normalize_entity_name("'Khovanskii condition'") == "Khovanskii condition"


def test_format_graph_context_for_prompt():
    gs = GraphSearch()
    data = [
        {
            "source": "Mutsuo Oka",
            "source_type": "Author",
            "relation": "WROTE",
            "target": "oka_1990.pdf",
            "target_type": "Paper",
        },
        {
            "source": "oka_1990.pdf",
            "source_type": "Paper",
            "relation": "DISCUSSES",
            "target": "Khovanskii non-degeneracy",
            "target_type": "Concept",
        },
    ]
    formatted = gs.format_graph_context_for_prompt(data)
    assert "(Author) Mutsuo Oka -> [wrote] -> (Paper) oka_1990.pdf" in formatted
    assert "(Paper) oka_1990.pdf -> [discusses] -> (Concept) Khovanskii non-degeneracy" in formatted


def test_graph_builder_and_search_live():
    client = get_neo4j_client()
    if not client.verify():
        return  # Skip if Neo4j is not available

    builder = GraphBuilder(client)
    gs = GraphSearch(client)

    extraction = GraphExtractionResult(
        paper_title="Test Singularity Paper",
        paper_filename="test_singularity.pdf",
        entities=[
            ExtractedEntity(name="Newton Polyhedron", type="Concept", description="Geometric polytope"),
            ExtractedEntity(name="Oka Theorem", type="Theorem", description="Non-degeneracy theorem"),
        ],
        relations=[
            ExtractedRelation(
                source="test_singularity.pdf",
                source_type="Paper",
                relation="DISCUSSES",
                target="Newton Polyhedron",
                target_type="Concept",
            ),
            ExtractedRelation(
                source="test_singularity.pdf",
                source_type="Paper",
                relation="PROVES",
                target="Oka Theorem",
                target_type="Theorem",
            ),
        ],
    )

    counts = builder.build_from_extraction(extraction, authors=["Mutsuo Oka"])
    assert counts["entities"] == 2
    assert counts["relations"] == 2

    # Query concepts by paper
    concepts = gs.find_concepts_by_paper("test_singularity.pdf")
    concept_names = [c["name"] for c in concepts]
    assert "Newton Polyhedron" in concept_names or "Oka Theorem" in concept_names

    # Clean up test nodes
    client.run("MATCH (p:Paper {filename: 'test_singularity.pdf'}) DETACH DELETE p")
    client.run("MATCH (c:Concept {name: 'Newton Polyhedron'}) DETACH DELETE c")
    client.run("MATCH (t:Theorem {name: 'Oka Theorem'}) DETACH DELETE t")
    client.run("MATCH (a:Author {name: 'Mutsuo Oka'}) DETACH DELETE a")
