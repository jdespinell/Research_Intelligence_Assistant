"""
Tests for retrieval formatting and vector search helpers.
"""

from src.retrieval.vector_search import SearchResult, format_chunks_for_prompt


def test_format_chunks_for_prompt():
    chunks = [
        SearchResult(
            chunk_id=1,
            paper_id=10,
            paper_title="On Khovanskii Non-degeneracy",
            paper_filename="khovanskii.pdf",
            chunk_index=0,
            page_number=3,
            content="This theorem proves non-degeneracy under certain conditions.",
            score=0.9123,
        ),
        SearchResult(
            chunk_id=2,
            paper_id=11,
            paper_title="Topological Triviality of Families",
            paper_filename="topological.pdf",
            chunk_index=1,
            page_number=7,
            content="Families of isolated singularities admit trivial fibrations.",
            score=0.8456,
        ),
    ]

    formatted = format_chunks_for_prompt(chunks)
    assert "[1] Source: On Khovanskii Non-degeneracy (p. 3)" in formatted
    assert "This theorem proves non-degeneracy" in formatted
    assert "[2] Source: Topological Triviality of Families (p. 7)" in formatted


def test_format_chunks_for_prompt_empty():
    assert format_chunks_for_prompt([]) == ""
