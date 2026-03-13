"""
src/ingestion/text_normalizer.py
─────────────────────────────────
Orchestrates the full normalisation flow for a document: it takes the raw
ExtractionResult (from pdf_extractor.py, with OCR-filled pages from
ocr_fallback.py already merged in) and produces the final `text_clean`
string that gets stored in the `documents_raw` table.

Two outputs are produced per document (mirroring the warehouse schema):
  - text_full   : original extracted text, lightly cleaned (no Arabic transforms)
  - text_clean  : fully normalised Arabic text used as the v2 pipeline input
"""

import logging
import re

from src.ingestion.pdf_extractor import ExtractionResult
from src.utils.arabic_utils import normalize_arabic, remove_extraction_noise

logger = logging.getLogger(__name__)

# Regex that matches the form-feed separator inserted between pages by
# ExtractionResult.text_full — used when we need to re-split later.
PAGE_SEPARATOR = "\f"


class TextNormalizer:
    """
    Produces clean, normalised Arabic text from a raw ExtractionResult.

    This class is intentionally thin — the heavy lifting lives in
    arabic_utils.py so it can be unit-tested in isolation.  The normalizer's
    job is simply to apply those utilities in the correct order and produce
    the two warehouse columns (text_full, text_clean) as a named tuple.
    """

    def normalise(self, result: ExtractionResult) -> tuple[str, str]:
        """
        Run the full normalisation pipeline over an ExtractionResult.

        Parameters
        ----------
        result : ExtractionResult
            The object returned by PDFExtractor.extract(), with OCR text
            already merged into pages that needed it (done by the
            orchestration layer in run_pipeline.py).

        Returns
        -------
        tuple[str, str]
            (text_full, text_clean)
            - text_full: concatenated raw text, lightly sanitised, suitable
              for archival and full-text search.
            - text_clean: fully normalised Arabic text — the primary input
              for every downstream stage.
        """
        # --- text_full: minimal cleaning, preserve as much original as possible
        raw_pages = [page.text_raw for page in result.pages]
        text_full = PAGE_SEPARATOR.join(raw_pages)
        text_full = self._light_clean(text_full)

        # --- text_clean: full Arabic normalisation
        text_clean = self._full_normalise(text_full)

        char_count_raw = len(text_full.replace(" ", "").replace("\n", ""))
        char_count_clean = len(text_clean.replace(" ", "").replace("\n", ""))
        logger.info(
            "Normalisation complete for '%s': raw=%d chars, clean=%d chars",
            result.s3_key,
            char_count_raw,
            char_count_clean,
        )

        return text_full, text_clean

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _light_clean(text: str) -> str:
        """
        Minimal sanitisation applied to text_full.
        We only remove clearly non-content characters — we do NOT normalise
        Arabic script here because text_full should remain as close to the
        original as possible for audit/archival purposes.
        """
        # Remove null bytes
        text = text.replace("\x00", "")
        # Collapse runs of more than 3 blank lines into 2
        text = re.sub(r"\n{4,}", "\n\n\n", text)
        # Trim trailing whitespace on each line
        text = "\n".join(line.rstrip() for line in text.splitlines())
        return text.strip()

    @staticmethod
    def _full_normalise(text: str) -> str:
        """
        Apply the full normalisation stack to produce text_clean.

        The order matters:
          1. remove_extraction_noise() strips PDF-specific artefacts first,
             before Arabic normalisation.  This way, noise patterns (like
             page numbers) are matched against unmolested characters.
          2. normalize_arabic() then handles the Arabic-script transforms:
             alef normalisation, harakat removal, digit conversion, etc.
        """
        # Preserve page separators through the process
        parts = text.split(PAGE_SEPARATOR)
        cleaned_parts = []
        for part in parts:
            part = remove_extraction_noise(part)
            part = normalize_arabic(part, aggressive=False)
            cleaned_parts.append(part)

        return PAGE_SEPARATOR.join(cleaned_parts)
