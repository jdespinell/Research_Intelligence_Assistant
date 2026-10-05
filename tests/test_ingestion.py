"""
Tests for ingestion components: chunker and embedder.
"""

import numpy as np
import pytest
from src.ingestion.chunker import Chunk, chunk_document, chunk_documents
from src.ingestion.embedder import embed_texts
from src.ingestion.pdf_loader import PageContent, PaperDocument


def test_chunk_document_basic():
    doc = PaperDocument(
        filename="test.pdf",
        filepath="/path/to/test.pdf",
        title="Test Doc",
        authors=["Author 1"],
        num_pages=1,
        pages=[PageContent(page_number=1, text="word " * 50)],
    )
    chunks = chunk_document(doc, chunk_size=10, chunk_overlap=2)
    assert len(chunks) >= 5
    for c in chunks:
        assert len(c.content.strip()) > 0
        assert c.paper_filename == "test.pdf"


def test_chunk_document_empty():
    doc = PaperDocument(
        filename="empty.pdf",
        filepath="/path/to/empty.pdf",
        title="Empty Doc",
        authors=[],
        num_pages=1,
        pages=[PageContent(page_number=1, text="   ")],
    )
    assert chunk_document(doc, chunk_size=100, chunk_overlap=20) == []


def test_chunk_document_multiple_pages():
    doc = PaperDocument(
        filename="sample.pdf",
        filepath="/path/to/sample.pdf",
        title="Sample Geometry Paper",
        authors=["A. Mathematician", "B. Geometer"],
        num_pages=2,
        pages=[
            PageContent(
                page_number=1,
                text="Introduction to singularity theory. " * 30,
            ),
            PageContent(
                page_number=2,
                text="Non-degeneracy and Newton polyhedra. " * 30,
            ),
        ],
    )
    chunks = chunk_document(doc, chunk_size=50, chunk_overlap=10)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)
    assert all(c.paper_filename == "sample.pdf" for c in chunks)
    assert all(c.page_number in [1, 2] for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_embed_texts():
    texts = [
        "Khovanskii non-degeneracy condition for singularity theory.",
        "Milnor fibration of isolated hypersurface singularities.",
    ]
    embeddings = embed_texts(texts)
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] == 768  # BAAI/bge-base-en-v1.5 dimension

    # Vectors should be L2-normalized (norm ~ 1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-5)
