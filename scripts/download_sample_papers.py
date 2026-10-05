"""
scripts/download_sample_papers.py

Utility to download sample open-access papers from arXiv (math.AG / Singularity Theory)
into data/papers/ for testing the ingestion and retrieval pipeline.

Usage:
    python scripts/download_sample_papers.py --max-papers 3
"""

import argparse
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from loguru import logger

PAPERS = [
    # Prominent open access arXiv papers touching singularity theory / Newton polyhedra / Milnor fibrations
    {"id": "0704.0924", "title": "Newton non-degeneracy of polynomial mappings"},
    {"id": "1103.3547", "title": "On Khovanskii non-degeneracy and Milnor numbers"},
    {"id": "1803.04780", "title": "Topological triviality of families of singularities"},
]


def download_arxiv_pdf(arxiv_id: str, dest_path: Path) -> bool:
    """Download a PDF from arXiv given an arXiv ID."""
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "ResearchIntelligenceAssistant/1.0 (academic research)"},
    )
    logger.info(f"Downloading arXiv:{arxiv_id} from {url}...")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            if content.startswith(b"%PDF"):
                dest_path.write_bytes(content)
                logger.success(f"Saved: {dest_path.name} ({len(content) // 1024} KB)")
                return True
            else:
                logger.warning(f"Response was not a valid PDF for {arxiv_id}")
                return False
    except Exception as e:
        logger.error(f"Failed to download arXiv:{arxiv_id}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download sample papers from arXiv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/papers"),
        help="Destination directory for PDF files",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=3,
        help="Maximum number of sample papers to download",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for p in PAPERS[: args.max_papers]:
        filename = f"arxiv_{p['id'].replace('/', '_')}.pdf"
        target = args.output_dir / filename
        if target.exists():
            logger.info(f"Already exists: {filename}")
            downloaded += 1
            continue

        if download_arxiv_pdf(p["id"], target):
            downloaded += 1
            time.sleep(2)  # Respect arXiv rate limit

    logger.info("=" * 50)
    logger.success(f"Ready: {downloaded} papers available in {args.output_dir}")


if __name__ == "__main__":
    main()
