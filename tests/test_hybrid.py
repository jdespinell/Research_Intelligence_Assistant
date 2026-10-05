"""
Tests for Hybrid GraphRAG router and retriever.
"""

from src.retrieval.hybrid import classify_query, extract_potential_entities


def test_classify_query():
    # Relational queries
    assert classify_query("¿Qué relación existe entre Newton polyhedra y Khovanskii non-degeneracy?") == "relational"
    assert classify_query("¿Qué autores conectan estos dos temas?") == "relational"
    assert classify_query("Find the path between Oka theorem and topological triviality") == "relational"
    assert classify_query("Who wrote the paper on non-degeneracy?") == "relational"

    # Factual queries
    assert classify_query("What is the definition of Khovanskii non-degeneracy?") == "factual"
    assert classify_query("¿Cuál es el enunciado del teorema de Oka?") == "factual"
    assert classify_query("Formula for the Milnor number") == "factual"

    # Hybrid queries (contains both aspects or complex)
    assert classify_query("What is Khovanskii non-degeneracy and what is the relationship with Milnor number?") == "hybrid"


def test_extract_potential_entities():
    q1 = 'What is "Khovanskii non-degeneracy" in algebraic geometry?'
    ents1 = extract_potential_entities(q1)
    assert any("Khovanskii non-degeneracy" in e for e in ents1)

    q2 = "Does Mutsuo Oka connect Newton Polyhedron to Milnor Fibration?"
    ents2 = extract_potential_entities(q2)
    assert any("Newton Polyhedron" in e or "Milnor Fibration" in e or "Oka" in e for e in ents2)
