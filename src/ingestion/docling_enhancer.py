"""
src/ingestion/docling_enhancer.py
──────────────────────────────────
Docling-based document enhancement for deep structural analysis.

Docling (IBM Research, MIT license) is used as an asynchronous post-processing
step that enriches documents already ingested by the primary PyMuPDF pipeline.
It adds capabilities that regex-based extraction cannot provide:

  - Table extraction with cell-level structure (tariff schedules, fee tables)
  - Cross-reference identification between articles and laws
  - Reliable reading order recovery on complex multi-column layouts
  - Vision-based extraction immune to font encoding issues

Architecture: Docling runs on an EC2 instance (g4dn.xlarge recommended) as a
separate asynchronous job triggered after the primary pipeline completes.
It reads from `documents_raw`, produces enriched `document_nodes` entries,
and flags documents for re-embedding in the OpenSearch index.

Deployment:
    pip install docling docling-core
    python -m src.ingestion.docling_enhancer --document-id <uuid>

Note: Arabic support in Docling is experimental as of 2025. Validate output
quality on the reference PDFs before enabling in production.
"""

import json
import logging
from pathlib import Path
from typing import Optional
from uuid import UUID

logger = logging.getLogger(__name__)


class DoclingEnhancer:
    """
    Wraps Docling's DocumentConverter to produce enriched structural output
    from Arabic legal PDFs, then updates the warehouse with the results.

    This class is intentionally decoupled from the main pipeline and is
    invoked as a separate async step to avoid blocking ingestion throughput.
    """

    def __init__(self) -> None:
        self._converter = None  # Lazy-loaded to avoid import cost in Lambda

    def _get_converter(self):
        """Lazy-load Docling to avoid cold-start penalty in environments
        where Docling is not installed (e.g. the primary Lambda function)."""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
                from docling.datamodel.pipeline_options import PipelineOptions

                pipeline_options = PipelineOptions()
                pipeline_options.do_ocr = True
                pipeline_options.do_table_structure = True
                # Arabic OCR — requires tesseract-lang-ara or equivalent
                pipeline_options.ocr_options.lang = ["ar"]

                self._converter = DocumentConverter(
                    pipeline_options=pipeline_options
                )
                logger.info("Docling DocumentConverter initialised.")
            except ImportError:
                raise RuntimeError(
                    "Docling is not installed. Run: pip install docling docling-core"
                )
        return self._converter

    def enhance_from_bytes(self, pdf_bytes: bytes, document_id: UUID) -> dict:
        """
        Run Docling over a PDF byte stream and return an enriched structural
        representation suitable for upserting into document_nodes.

        Parameters
        ----------
        pdf_bytes : bytes
            Raw PDF content.
        document_id : UUID
            The warehouse document ID this enhancement belongs to.

        Returns
        -------
        dict
            Enriched document structure with keys:
              - nodes: list of enriched node dicts
              - tables: list of extracted table dicts
              - cross_references: list of article cross-reference pairs
              - confidence: overall extraction confidence (0-1)
        """
        import tempfile

        converter = self._get_converter()

        # Docling requires a file path — write to a temp file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = Path(tmp.name)

        try:
            result = converter.convert(str(tmp_path))
            doc = result.document

            nodes = self._extract_nodes(doc, document_id)
            tables = self._extract_tables(doc)
            cross_refs = self._extract_cross_references(doc)

            logger.info(
                "Docling enhancement for %s: %d nodes, %d tables, %d cross-refs",
                document_id, len(nodes), len(tables), len(cross_refs),
            )

            return {
                "document_id": str(document_id),
                "nodes": nodes,
                "tables": tables,
                "cross_references": cross_refs,
                "confidence": self._compute_confidence(doc),
            }
        finally:
            tmp_path.unlink(missing_ok=True)

    def enhance_from_s3(
        self,
        s3_key: str,
        document_id: UUID,
        bucket: Optional[str] = None,
    ) -> dict:
        """
        Download a PDF from S3 and run Docling enhancement on it.
        Convenience wrapper around enhance_from_bytes for pipeline integration.
        """
        from src.ingestion.s3_retriever import S3Retriever

        retriever = S3Retriever()
        pdf_bytes = retriever.download_to_bytes(s3_key)
        return self.enhance_from_bytes(pdf_bytes, document_id)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_nodes(doc, document_id: UUID) -> list[dict]:
        """
        Convert Docling's document structure into a list of node dicts
        compatible with the document_nodes warehouse schema.
        """
        nodes = []
        try:
            for item in doc.texts:
                label = str(item.label) if hasattr(item, "label") else "body"
                node_type = "article" if "heading" in label.lower() else "body"
                nodes.append({
                    "document_id": str(document_id),
                    "node_type": node_type,
                    "text_content": item.text,
                    "heading": item.text if "heading" in label.lower() else None,
                    "extra": {"docling_label": label, "source": "docling"},
                })
        except Exception as exc:
            logger.warning("Docling node extraction partial failure: %s", exc)
        return nodes

    @staticmethod
    def _extract_tables(doc) -> list[dict]:
        """Extract tables from Docling output as structured dicts."""
        tables = []
        try:
            for i, table in enumerate(doc.tables):
                rows = []
                if hasattr(table, "data") and table.data:
                    for row in table.data.grid:
                        rows.append([cell.text if cell else "" for cell in row])
                tables.append({
                    "table_index": i,
                    "rows": rows,
                    "caption": str(table.caption_text) if hasattr(table, "caption_text") else None,
                })
        except Exception as exc:
            logger.warning("Docling table extraction partial failure: %s", exc)
        return tables

    @staticmethod
    def _extract_cross_references(doc) -> list[dict]:
        """
        Identify cross-references between articles using Docling's text output.
        Looks for patterns like 'المادة (X) من هذا القانون' or law number references.
        """
        import re
        cross_refs = []
        XREF_PATTERN = re.compile(
            r"(المادة\s*\(\d+\)|القانون الاتحادي رقم\s*\(\d+\)\s*لسنة\s*\d{4})",
            re.UNICODE,
        )
        try:
            for item in doc.texts:
                matches = XREF_PATTERN.findall(item.text)
                for match in matches:
                    cross_refs.append({
                        "source_text": item.text[:100],
                        "reference": match,
                    })
        except Exception as exc:
            logger.warning("Docling cross-reference extraction partial failure: %s", exc)
        return cross_refs

    @staticmethod
    def _compute_confidence(doc) -> float:
        """Estimate overall extraction confidence from Docling output."""
        try:
            total = len(list(doc.texts)) if doc.texts else 0
            return min(1.0, total / 50.0)  # 50+ text blocks = high confidence
        except Exception:
            return 0.5
PYEOF