"""
src/ingestion/pdf_extractor.py
───────────────────────────────
Primary text extraction layer using PyMuPDF (imported as `fitz`).

PyMuPDF is chosen as the primary extractor because UAE government PDFs store
Arabic glyphs using a custom font encoding (Arabic Presentation Forms,
U+FE70–U+FEFF). PyMuPDF's PDFium engine correctly resolves these to canonical
Unicode Arabic (U+0600–U+06FF) via the PDF's ToUnicode CMap, enabling all
downstream regex matching, structural extraction, and embedding generation.

When a page does not yield sufficient Arabic text — e.g., because the PDF
was created from a scanned image — the extractor signals this via the
`needs_ocr` flag per page, and the OCR fallback module (Amazon Textract)
takes over for those pages only.

For documents requiring deep structural analysis beyond regex-based extraction
(table extraction, cross-reference identification, complex multi-column layouts),
see the Docling integration stub in `src/ingestion/docling_enhancer.py`.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import fitz  # PyMuPDF

from config.settings import settings
from src.utils.arabic_utils import count_arabic_chars

logger = logging.getLogger(__name__)


@dataclass
class ExtractedPage:
    """Represents a single extracted page from the document."""
    page_number: int
    text_raw: str
    needs_ocr: bool = False
    image_bytes: Optional[bytes] = None


@dataclass
class ExtractionResult:
    """Full extraction result for a document."""
    s3_key: str
    pages: list[ExtractedPage] = field(default_factory=list)
    total_pages: int = 0
    pages_needing_ocr: int = 0

    @property
    def text_full(self) -> str:
        return "\f".join(p.text_raw for p in self.pages)

    @property
    def is_largely_native_text(self) -> bool:
        if not self.pages:
            return True
        return (self.pages_needing_ocr / len(self.pages)) < 0.30


class PDFExtractor:
    """
    Wraps PyMuPDF to extract per-page text from a PDF byte stream.
    Works entirely in-memory to avoid writing temporary files to disk,
    which matters for Lambda deployments.
    """

    def __init__(
        self,
        min_arabic_chars: int = settings.min_arabic_chars_for_direct_extract,
    ) -> None:
        self.min_arabic_chars = min_arabic_chars

    def extract(self, pdf_bytes: bytes, s3_key: str = "") -> ExtractionResult:
        """
        Extract text from a PDF byte stream page by page.

        For each page, PyMuPDF is attempted first. Pages returning fewer
        Arabic characters than `min_arabic_chars` are flagged for Textract OCR
        and rasterised to JPEG at 200 DPI for the fallback call.
        """
        result = ExtractionResult(s3_key=s3_key)

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        except Exception as exc:
            logger.error("PyMuPDF failed to open '%s': %s", s3_key, exc)
            raise

        result.total_pages = len(doc)
        logger.info("Processing PDF '%s': %d pages", s3_key, result.total_pages)

        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1
            raw_text = page.get_text("text")
            arabic_char_count = count_arabic_chars(raw_text)
            needs_ocr = arabic_char_count < self.min_arabic_chars

            extracted_page = ExtractedPage(
                page_number=page_number,
                text_raw=raw_text,
                needs_ocr=needs_ocr,
            )

            if needs_ocr:
                logger.debug(
                    "Page %d of '%s' needs OCR (%d Arabic chars found)",
                    page_number, s3_key, arabic_char_count,
                )
                extracted_page.image_bytes = self._rasterise_page(page)
                result.pages_needing_ocr += 1

            result.pages.append(extracted_page)

        doc.close()
        logger.info(
            "Extraction complete for '%s': %d/%d pages need OCR",
            s3_key, result.pages_needing_ocr, result.total_pages,
        )
        return result

    @staticmethod
    def _rasterise_page(page: fitz.Page, dpi: int = 200) -> bytes:
        """Rasterise a page to JPEG bytes for Amazon Textract OCR."""
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, clip=page.rect, colorspace=fitz.csRGB)
        return pix.tobytes("jpeg")
