"""
Text chunking strategy.

Splits a PaperDocument into overlapping text chunks.
Each chunk carries metadata: paper_id reference, chunk_index, page_number.
"""

from dataclasses import dataclass

from loguru import logger

from src.config import get_settings
from src.ingestion.pdf_loader import PaperDocument

settings = get_settings()


@dataclass
class Chunk:
    paper_filename: str   # for linking before DB insertion
    chunk_index: int      # 0-indexed
    page_number: int      # page where this chunk starts
    content: str          # text content of the chunk


def chunk_document(
    doc: PaperDocument,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """
    Split a PaperDocument into fixed-size overlapping text chunks.

    Strategy:
      1. Concatenate all pages preserving page boundaries.
      2. Split by words (not bytes) to avoid cutting mid-word.
      3. Slide a window of `chunk_size` words with `chunk_overlap` word overlap.

    Word-based chunking is more stable than character-based for scientific text
    that has variable-length mathematical expressions.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    # Build list of (word, page_number) pairs
    word_page_pairs: list[tuple[str, int]] = []
    for page in doc.pages:
        words = page.text.split()
        for word in words:
            word_page_pairs.append((word, page.page_number))

    if not word_page_pairs:
        logger.warning(f"No text extracted from '{doc.filename}' — skipping chunking")
        return []

    chunks: list[Chunk] = []
    start = 0
    chunk_idx = 0

    while start < len(word_page_pairs):
        end = min(start + chunk_size, len(word_page_pairs))
        words_slice = [wp[0] for wp in word_page_pairs[start:end]]
        page_num = word_page_pairs[start][1]  # page of the first word in chunk

        content = " ".join(words_slice).strip()
        if content:
            chunks.append(
                Chunk(
                    paper_filename=doc.filename,
                    chunk_index=chunk_idx,
                    page_number=page_num,
                    content=content,
                )
            )
            chunk_idx += 1

        if end == len(word_page_pairs):
            break

        # Slide window with overlap
        start = end - chunk_overlap

    logger.debug(
        f"Chunked '{doc.filename}' → {len(chunks)} chunks "
        f"(size={chunk_size} words, overlap={chunk_overlap})"
    )
    return chunks


def chunk_documents(docs: list[PaperDocument], **kwargs) -> list[Chunk]:
    """Chunk a list of documents."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, **kwargs))
    return all_chunks
