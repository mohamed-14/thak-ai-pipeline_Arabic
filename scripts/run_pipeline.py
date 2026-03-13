"""
scripts/run_pipeline.py
────────────────────────
CLI entrypoint for the ThakAI Document ETL Pipeline.

This script is the top-level orchestrator: it wires together all the
pipeline stages in the correct order and provides a command-line interface
for running single-document or batch jobs.

Usage examples
──────────────
# Initialise the database schema (run once):
    python scripts/run_pipeline.py --init-db

# Process all Arabic laws in the S3 bucket:
    python scripts/run_pipeline.py --lang arabic --doc-type Laws

# Process a single document by S3 key:
    python scripts/run_pipeline.py --s3-key Laws/arabic/federal_law_01_2024.pdf

# Process all Arabic documents (laws + regulatory):
    python scripts/run_pipeline.py --lang arabic

Design: the orchestrator catches and logs errors per document so that a
single bad PDF does not abort the entire batch.  Failed documents are
marked with status="error" in the warehouse for later inspection and retry.
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the project root without installing as a package
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.progress import track

from config.settings import settings
from src.database.models import Base
from src.database.warehouse import WarehouseClient, get_session, _engine
from src.ingestion.ocr_fallback import TextractOCR
from src.ingestion.pdf_extractor import PDFExtractor
from src.ingestion.s3_retriever import S3Retriever
from src.ingestion.text_normalizer import TextNormalizer
from src.pipeline.document_profiler import DocumentProfiler
from src.pipeline.metadata_extractor import MetadataExtractor
from src.pipeline.page_splitter import PageSplitter
from src.pipeline.structural_extractor import StructuralExtractor

console = Console()
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("run_pipeline")


# ── Pipeline orchestration ────────────────────────────────────────────────────

def process_document(s3_key: str) -> None:
    """
    Run the full pre-processing + v2 pipeline for a single S3 document.

    This function is intentionally procedural rather than object-oriented —
    it reads like a recipe, which makes it easy to follow the data flow:
    S3 → bytes → extraction → normalisation → DB → stages 1-4.

    Parameters
    ----------
    s3_key : str
        The S3 object key of the PDF to process.
    """
    console.print(f"\n[bold cyan]Processing:[/bold cyan] {s3_key}")

    # Initialise shared collaborators
    warehouse = WarehouseClient()
    retriever = S3Retriever()
    extractor = PDFExtractor()
    ocr = TextractOCR()
    normalizer = TextNormalizer()
    splitter = PageSplitter(warehouse)
    profiler = DocumentProfiler(warehouse)
    struct_extractor = StructuralExtractor(warehouse)
    meta_extractor = MetadataExtractor(warehouse)

    # ── Pre-Processing Stage ──────────────────────────────────────────────────

    # 1. Download from S3
    console.print("  [dim]→ Downloading from S3...[/dim]")
    pdf_bytes = retriever.download_to_bytes(s3_key)
    s3_meta = retriever.get_object_metadata(s3_key)

    # 2. Extract text with PyMuPDF
    console.print("  [dim]→ Extracting text (PyMuPDF)...[/dim]")
    extraction_result = extractor.extract(pdf_bytes, s3_key=s3_key)

    # 3. OCR fallback for pages that PyMuPDF couldn't read
    if extraction_result.pages_needing_ocr > 0:
        console.print(
            f"  [yellow]→ Running OCR on {extraction_result.pages_needing_ocr} "
            f"pages (Amazon Textract)...[/yellow]"
        )
        for page in extraction_result.pages:
            if page.needs_ocr and page.image_bytes:
                page.text_raw = ocr.ocr_page_image(page.image_bytes)
                page.needs_ocr = False   # Mark as resolved
                page.image_bytes = None  # Free memory

    # 4. Normalise text
    console.print("  [dim]→ Normalising Arabic text...[/dim]")
    text_full, text_clean = normalizer.normalise(extraction_result)

    # 5. Store in documents_raw
    doc = warehouse.upsert_document_raw(
        file_path=s3_key,
        text_full=text_full,
        text_clean=text_clean,
        language="arabic",
        s3_etag=s3_meta.get("etag"),
        jurisdiction="UAE",  # Default; Stage 4 will refine this
    )
    console.print(f"  [green]✓ Pre-processing complete[/green] (document id: {doc.id})")

    # ── v2 Pipeline ───────────────────────────────────────────────────────────

    # Stage 1: Page Splitting
    console.print("  [dim]→ Stage 1: Page splitting...[/dim]")
    pages = splitter.split_and_store(doc.id, text_clean)
    console.print(f"  [green]✓ Stage 1:[/green] {len(pages)} pages")

    # Stage 2: Document Profiling
    console.print("  [dim]→ Stage 2: Document profiling (Bedrock LLM)...[/dim]")
    profile = profiler.profile(doc.id, text_clean)
    hierarchy = profile.get("hierarchy", [])
    console.print(f"  [green]✓ Stage 2:[/green] hierarchy = {hierarchy}")

    # Stage 3: Structural Extraction
    console.print("  [dim]→ Stage 3: Structural extraction...[/dim]")
    nodes = struct_extractor.extract_and_store(doc.id, text_clean, profile)
    console.print(f"  [green]✓ Stage 3:[/green] {len(nodes)} nodes extracted")

    # Stage 4: Metadata Extraction
    console.print("  [dim]→ Stage 4: Metadata extraction (Bedrock LLM)...[/dim]")
    metadata = meta_extractor.extract_and_store(doc.id, text_clean)
    console.print(
        f"  [green]✓ Stage 4:[/green] "
        f"{metadata.get('document_type')} — {metadata.get('law_number')}"
    )

    # Mark the full pipeline as complete
    warehouse.mark_document_status(doc.id, "complete")
    console.print(f"  [bold green]✓ Document pipeline complete:[/bold green] {s3_key}\n")


def run_batch(language: str = "arabic", doc_type: str | None = None) -> None:
    """
    Process all documents in the S3 bucket matching the given language / type.
    """
    retriever = S3Retriever()
    doc_types = [doc_type] if doc_type else ["Laws", "Regulatory"]

    all_keys: list[str] = []
    for dtype in doc_types:
        all_keys.extend(list(retriever.list_documents(language=language, doc_type=dtype)))

    if not all_keys:
        console.print(f"[yellow]No {language} documents found in S3 bucket.[/yellow]")
        return

    console.print(f"\n[bold]Found {len(all_keys)} documents to process.[/bold]")
    errors: list[tuple[str, str]] = []

    for key in track(all_keys, description="Processing documents..."):
        try:
            process_document(key)
        except Exception as exc:
            logger.error("Failed to process '%s': %s", key, exc, exc_info=True)
            errors.append((key, str(exc)))
            # Mark document as errored in DB if it was already created
            warehouse = WarehouseClient()
            doc = warehouse.get_document_by_path(key)
            if doc:
                warehouse.mark_document_status(doc.id, "error", error_message=str(exc))

    console.print(
        f"\n[bold green]Batch complete.[/bold green] "
        f"{len(all_keys) - len(errors)} succeeded, {len(errors)} failed."
    )
    if errors:
        console.print("[red]Failed documents:[/red]")
        for key, err in errors:
            console.print(f"  {key}: {err}")


# ── CLI argument parsing ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ThakAI Document ETL Pipeline — Arabic Legal Documents"
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialise the database schema and exit.",
    )
    parser.add_argument(
        "--s3-key",
        type=str,
        default=None,
        help="Process a single document by its S3 key.",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="arabic",
        choices=["arabic", "english"],
        help="Language prefix to process in batch mode (default: arabic).",
    )
    parser.add_argument(
        "--doc-type",
        type=str,
        default=None,
        choices=["Laws", "Regulatory"],
        help="Document type to process in batch mode. Omit to process both.",
    )

    args = parser.parse_args()

    if args.init_db:
        Base.metadata.create_all(_engine)
        console.print("[bold green]Database schema initialised.[/bold green]")
        return

    if args.s3_key:
        process_document(args.s3_key)
    else:
        run_batch(language=args.lang, doc_type=args.doc_type)


if __name__ == "__main__":
    main()
