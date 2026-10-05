"""
PDF text extraction using PyMuPDF (fitz).

For each PDF, returns a list of Page objects with:
  - page_number (1-indexed)
  - text content
  - basic metadata extracted from the document
"""

from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (avoid deprecated `fitz` alias)
from loguru import logger


@dataclass
class PageContent:
    page_number: int   # 1-indexed
    text: str


@dataclass
class PaperDocument:
    filename: str          # original PDF filename
    filepath: str          # absolute path
    title: str             # from PDF metadata or filename fallback
    authors: list[str]     # from PDF metadata (may be empty)
    num_pages: int
    pages: list[PageContent] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


def extract_pdf(pdf_path: Path) -> PaperDocument:
    """
    Extract text and metadata from a single PDF file.
    Returns a PaperDocument with one PageContent per page.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    meta = doc.metadata or {}

    # Try to get title from metadata; fall back to filename stem
    title = (meta.get("title", "").strip() or pdf_path.stem.replace("_", " ").replace("-", " ")).replace("\x00", "")

    # Authors: metadata field is a single string — split by common separators
    authors_raw = meta.get("author", "").strip().replace("\x00", "")
    if authors_raw:
        import re
        authors = [a.strip().replace("\x00", "") for a in re.split(r"[;,&]|\band\b", authors_raw) if a.strip()]
    else:
        authors = []

    total_pages = len(doc)
    pages: list[PageContent] = []
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text").replace("\x00", "")  # plain text extraction without NUL bytes
        # Skip nearly empty pages (e.g. cover images, blank pages)
        if len(text.strip()) > 50:
            pages.append(PageContent(page_number=page_num + 1, text=text.strip()))

    doc.close()

    logger.debug(
        f"Extracted '{pdf_path.name}' → {len(pages)} pages "
        f"(title: '{title[:50]}', authors: {authors})"
    )

    return PaperDocument(
        filename=pdf_path.name,
        filepath=str(pdf_path.resolve()),
        title=title,
        authors=authors,
        num_pages=total_pages,
        pages=pages,
    )


def iter_papers_dir(papers_dir: Path):
    """Yield PaperDocument one by one from a directory. Skips files that fail to parse."""
    papers_dir = Path(papers_dir)
    pdf_files = sorted(papers_dir.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF(s) in {papers_dir}")

    for idx, pdf_path in enumerate(pdf_files, 1):
        try:
            yield extract_pdf(pdf_path)
        except Exception as e:
            logger.warning(f"Skipping '{pdf_path.name}': {e}")


def load_papers_dir(papers_dir: Path) -> list[PaperDocument]:
    """Load all PDFs from a directory into a list."""
    docs = list(iter_papers_dir(papers_dir))
    logger.success(f"Loaded {len(docs)} PDFs successfully")
    return docs
