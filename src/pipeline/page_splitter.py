"""
src/pipeline/page_splitter.py
──────────────────────────────
Stage 1 of the v2 Document Processing Pipeline: Page Splitting.

Goal: segment the normalised document text into logical page units that
align with the original PDF page layout.  The result is stored in the
`document_pages` table.

Why this matters: downstream stages (especially structural extraction)
benefit from knowing page boundaries because legal documents often start
a new chapter or article at the top of a page.  This helps the LLM
profiling step focus on representative pages rather than arbitrary slices.

Two splitting strategies are implemented:
  1. `marker_detection`  — preferred.  Uses the form-feed (\f) characters
     that PDFExtractor inserts between pages.  These are real page boundaries
     directly derived from the PDF structure.
  2. `length_heuristic`  — fallback.  If no form-feeds exist (e.g., the text
     was produced by Textract's async mode which doesn't insert them), the
     text is split into estimated pages based on character count.
"""

import logging
import re
from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from src.database.warehouse import WarehouseClient

logger = logging.getLogger(__name__)

# Characters per "logical page" when using the length heuristic.
# Arabic legal pages at standard font sizes typically contain 1,800–2,500
# characters.  We use 2,000 as a conservative estimate.
_CHARS_PER_LOGICAL_PAGE = 2_000

# Form-feed character inserted between pages by PDFExtractor
_PAGE_SEP = "\f"

SplitStrategy = Literal["marker_detection", "length_heuristic"]


@dataclass
class SplitPage:
    """Internal representation of a split page before DB storage."""
    page_number: int
    text_content: str
    strategy_used: SplitStrategy
    page_boundary_type: str      # "real" or "logical"
    corresponds_to_pdf_page: bool


class PageSplitter:
    """
    Splits a document's clean text into page segments and persists them
    to the `document_pages` warehouse table.
    """

    def __init__(self, warehouse: WarehouseClient) -> None:
        self._db = warehouse

    def split_and_store(self, document_id: UUID, text_clean: str) -> list[SplitPage]:
        """
        Detect page boundaries in `text_clean` and persist to the database.

        Parameters
        ----------
        document_id : UUID
            The ID of the document_raw row this belongs to.
        text_clean : str
            Normalised document text (text_clean from documents_raw).

        Returns
        -------
        list[SplitPage]
            The list of split pages (also written to the DB as a side-effect).
        """
        # Choose strategy based on whether form-feed markers are present
        if _PAGE_SEP in text_clean:
            pages = self._split_by_marker(text_clean)
            logger.info(
                "Document %s: split into %d real pages via marker_detection",
                document_id,
                len(pages),
            )
        else:
            pages = self._split_by_length(text_clean)
            logger.info(
                "Document %s: no page markers found; split into %d logical pages "
                "via length_heuristic",
                document_id,
                len(pages),
            )

        # Persist to the warehouse
        page_dicts = [
            {
                "page_number": p.page_number,
                "text_content": p.text_content,
                "strategy_used": p.strategy_used,
                "page_boundary_type": p.page_boundary_type,
                "corresponds_to_pdf_page": p.corresponds_to_pdf_page,
                "extra": {
                    "char_count": len(p.text_content),
                    "total_pages": len(pages),
                },
            }
            for p in pages
            if p.text_content.strip()  # skip blank pages
        ]
        self._db.store_pages(document_id, page_dicts)

        return pages

    # ── Strategy implementations ──────────────────────────────────────────────

    @staticmethod
    def _split_by_marker(text: str) -> list[SplitPage]:
        """
        Split on form-feed characters (\f) that PDFExtractor inserts at
        each PDF page boundary.  These are real page boundaries.
        """
        raw_pages = text.split(_PAGE_SEP)
        return [
            SplitPage(
                page_number=i + 1,
                text_content=raw_page.strip(),
                strategy_used="marker_detection",
                page_boundary_type="real",
                corresponds_to_pdf_page=True,
            )
            for i, raw_page in enumerate(raw_pages)
        ]

    @staticmethod
    def _split_by_length(text: str) -> list[SplitPage]:
        """
        When no form-feed markers exist, estimate page boundaries by
        character count.  A new logical page starts every `_CHARS_PER_LOGICAL_PAGE`
        characters, but we try to break at paragraph boundaries (double
        newlines) to avoid splitting mid-sentence.
        """
        paragraphs = re.split(r"\n{2,}", text)
        pages: list[SplitPage] = []
        current_page_parts: list[str] = []
        current_length = 0
        page_number = 1

        for para in paragraphs:
            para_len = len(para)
            if current_length + para_len > _CHARS_PER_LOGICAL_PAGE and current_page_parts:
                # Flush the current page
                pages.append(
                    SplitPage(
                        page_number=page_number,
                        text_content="\n\n".join(current_page_parts).strip(),
                        strategy_used="length_heuristic",
                        page_boundary_type="logical",
                        corresponds_to_pdf_page=False,
                    )
                )
                page_number += 1
                current_page_parts = [para]
                current_length = para_len
            else:
                current_page_parts.append(para)
                current_length += para_len

        # Flush the last page
        if current_page_parts:
            pages.append(
                SplitPage(
                    page_number=page_number,
                    text_content="\n\n".join(current_page_parts).strip(),
                    strategy_used="length_heuristic",
                    page_boundary_type="logical",
                    corresponds_to_pdf_page=False,
                )
            )

        return pages
