"""PDF → chunk → embed → Supabase upsert pipeline.

Usage:
  python scripts/ingest_pdf.py path/to/book.pdf --category textbook_passages --difficulty TOPIK_1

This script is a thin wrapper over the existing ingestion infrastructure:
  - PdfIngester (app.ingest.ingesters.pdf)      — pdfplumber page extraction
  - PageBoundaryChunker (app.ingest.chunkers)    — PDF-appropriate chunking
  - build_embedder (app.ai.factory)              — provider-agnostic embedder
  - build_store (app.storage.stores)             — supabase / sqlite_vec / pinecone
  - IngestionPipeline (app.ingest.pipeline)      — chunk → embed → upsert

The --difficulty value is injected into the metadata of every chunk so it can
be filtered at retrieval time (e.g. TOPIK_1, TOPIK_2, etc.).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.ai.factory import build_embedder
from app.ingest.ingesters.pdf import PdfIngester
from app.ingest.pipeline import IngestionPipeline
from app.storage.stores import build_store


class _DifficultyPdfIngester(PdfIngester):
    """Thin subclass that stamps difficulty into every Document's metadata."""

    def __init__(self, paths, *, category: str, difficulty: str) -> None:
        super().__init__(paths, category=category)
        self._difficulty = difficulty

    def _build_document(self, path: Path):
        doc = super()._build_document(path)
        if doc is not None:
            doc.metadata["difficulty"] = self._difficulty
        return doc


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest a single PDF into the RAG knowledge base.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file to ingest.")
    parser.add_argument(
        "--category",
        default="textbook_passages",
        help="Knowledge category tag stored with each chunk.",
    )
    parser.add_argument(
        "--difficulty",
        default="TOPIK_1",
        help="Difficulty level stamped into chunk metadata (e.g. TOPIK_1, TOPIK_2).",
    )
    args = parser.parse_args()

    pdf_path: Path = args.pdf_path
    if not pdf_path.exists():
        print(f"[error] PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    settings = get_settings()
    embedder = build_embedder(settings)
    store = build_store(settings)
    pipeline = IngestionPipeline(embedder=embedder, store=store)

    ingester = _DifficultyPdfIngester(
        [pdf_path],
        category=args.category,
        difficulty=args.difficulty,
    )

    print(f"[ingest_pdf] Starting ingestion: {pdf_path.name}")
    print(f"[ingest_pdf]   category   = {args.category}")
    print(f"[ingest_pdf]   difficulty = {args.difficulty}")

    stats = await pipeline.run(ingester)

    print(
        f"[ingest_pdf] Done — docs={stats.docs} chunks={stats.chunks} "
        f"skipped={stats.skipped} unchanged={stats.unchanged}"
    )


if __name__ == "__main__":
    asyncio.run(main())
