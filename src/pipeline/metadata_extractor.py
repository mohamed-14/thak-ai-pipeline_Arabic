"""
src/pipeline/metadata_extractor.py
────────────────────────────────────
Stage 4 of the v2 Pipeline: Metadata Extraction.

This final stage extracts structured legal metadata from the document
using a combination of:
  1. Regex-based pattern matching  — fast and reliable for well-formatted
     fields like law numbers and dates that follow predictable patterns.
  2. AWS Bedrock LLM call           — for fields that require contextual
     understanding, such as the document status (in_force vs. amended) and
     the issuing authority, which don't follow a single regex pattern.

The hybrid approach minimises LLM tokens while ensuring we handle the
long-tail of legal document formats that don't fit clean patterns.

Output: a row in `document_metadata` with all extracted fields, plus
an update to `documents_raw.title` for quick display queries.
"""

import json
import logging
import re
from typing import Optional
from uuid import UUID

import boto3
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from src.database.warehouse import WarehouseClient
from src.utils.arabic_utils import LAW_NUMBER_PATTERN

logger = logging.getLogger(__name__)


# ── Date patterns for UAE legal documents ────────────────────────────────────
# Dates appear in several formats in UAE legislation.
# "الموافق" means "corresponding to" and introduces the Gregorian date after
# a Hijri date, which is the reliable machine-readable one.

_DATE_GREGORIAN_PATTERN = re.compile(
    r"الموافق\s+(\d{1,2})[/\-\u060c،,\s]+(\d{1,2})[/\-\u060c،,\s]+(\d{4})",
    re.UNICODE,
)

_DATE_ISO_PATTERN = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

# Article 1 (or last article) often states the effective date
_EFFECTIVE_DATE_PATTERN = re.compile(
    r"(يُعمل|يسري|تسري|نفاذ|العمل)\s+.{0,60}(\d{4})",
    re.UNICODE | re.DOTALL,
)


# ── Bedrock metadata prompt ───────────────────────────────────────────────────

_METADATA_SYSTEM_PROMPT = """\
You are an expert in UAE and Arabic legal documents.
Extract structured metadata from the provided Arabic legal document text.
Respond ONLY with a valid JSON object, no markdown, no explanation.

Required JSON schema:
{
  "title": "<Full Arabic title of the document>",
  "title_en": "<English translation of the title>",
  "law_number": "<e.g. Federal Law No. 1 of 2024, or null>",
  "document_type": "<Law | Decree | Resolution | Regulation | Decision | Other>",
  "issuing_authority": "<The government body or ruler that issued this document>",
  "jurisdiction": "<UAE | Dubai | Abu Dhabi | Sharjah | etc.>",
  "issue_date": "<Date as it appears, or null>",
  "effective_date": "<Effective/entry-into-force date, or null>",
  "status": "<in_force | repealed | amended | draft | unknown>",
  "amends_document": "<Reference to a law this amends, or null>",
  "subject_area": "<e.g. Finance, Healthcare, Environment, Technology>",
  "extraction_confidence": <float between 0.0 and 1.0>
}
"""

_METADATA_USER_TEMPLATE = """\
Extract metadata from this Arabic legal document.

Document text (first 3000 characters):
---
{text_sample}
---
"""


class MetadataExtractor:
    """
    Extracts and persists legal metadata from the normalised document text.
    """

    def __init__(self, warehouse: WarehouseClient) -> None:
        self._db = warehouse
        self._bedrock = boto3.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
            **(
                {
                    "aws_access_key_id": settings.aws_access_key_id,
                    "aws_secret_access_key": settings.aws_secret_access_key,
                }
                if settings.aws_access_key_id
                else {}
            ),
        )

    def extract_and_store(self, document_id: UUID, text_clean: str) -> dict:
        """
        Run both regex and LLM metadata extraction, merge the results,
        and persist them to `document_metadata`.

        The regex pass runs first to cheaply capture high-confidence fields.
        The LLM call then fills in anything the regex couldn't find and
        provides English translation + contextual fields like status and
        subject area.

        Parameters
        ----------
        document_id : UUID
        text_clean : str
            Normalised full document text.

        Returns
        -------
        dict
            The merged metadata dict as it was written to the database.
        """
        # Pass 1: fast regex extraction
        regex_metadata = self._extract_via_regex(text_clean)

        # Pass 2: LLM extraction for remaining / uncertain fields
        llm_metadata = self._extract_via_llm(text_clean)

        # Merge: regex results take precedence for fields both found
        # (regex is more precise for structured fields like law numbers),
        # LLM fills gaps and provides contextual fields.
        merged = {**llm_metadata, **{k: v for k, v in regex_metadata.items() if v}}

        # Persist to the warehouse
        self._db.store_metadata(document_id, merged)
        self._db.mark_document_status(document_id, "metadata_extracted")

        logger.info(
            "Metadata extracted for document %s: type=%s, jurisdiction=%s, status=%s",
            document_id,
            merged.get("document_type"),
            merged.get("jurisdiction"),
            merged.get("status"),
        )
        return merged

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_via_regex(text: str) -> dict:
        """
        Extract metadata fields that follow predictable patterns.
        Regex is fast (< 1ms) and doesn't consume any API quota.
        """
        metadata: dict = {}

        # Law number (e.g., "قانون اتحادي رقم (3) لسنة 2022")
        law_ref = LAW_NUMBER_PATTERN.search(text)
        if law_ref:
            metadata["law_number"] = law_ref.group(0)
            doc_type_map = {
                "قانون": "Law",
                "مرسوم": "Decree",
                "قرار": "Resolution",
            }
            metadata["document_type"] = doc_type_map.get(law_ref.group(1), "Other")

        # Gregorian date (appears after Hijri date in most UAE laws)
        date_match = _DATE_GREGORIAN_PATTERN.search(text[:3000])
        if date_match:
            day, month, year = date_match.group(1), date_match.group(2), date_match.group(3)
            metadata["issue_date"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        # Effective date
        eff_match = _EFFECTIVE_DATE_PATTERN.search(text)
        if eff_match:
            metadata["effective_date"] = eff_match.group(2)

        # Jurisdiction detection
        if "اتحادي" in text[:2000] or "الدولة" in text[:2000]:
            metadata["jurisdiction"] = "UAE"
        elif "دبي" in text[:2000]:
            metadata["jurisdiction"] = "Dubai"
        elif "أبوظبي" in text[:2000] or "أبو ظبي" in text[:2000]:
            metadata["jurisdiction"] = "Abu Dhabi"

        return metadata

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True,
    )
    def _extract_via_llm(self, text: str) -> dict:
        """
        Use Bedrock to extract metadata fields requiring contextual understanding.
        We only send the first 3,000 characters — the header and preamble
        contain virtually all metadata fields and staying short saves tokens.
        """
        text_sample = text[:3_000]
        user_message = _METADATA_USER_TEMPLATE.format(text_sample=text_sample)

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1_024,   # Metadata is compact; 1k tokens is plenty
            "system": _METADATA_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
        }

        response = self._bedrock.invoke_model(
            modelId=settings.bedrock_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )

        body = json.loads(response["body"].read())
        raw_text = body["content"][0]["text"].strip()

        # Strip markdown fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.error("Bedrock metadata response was not valid JSON: %s", raw_text[:300])
            raise ValueError(f"Bedrock metadata response was not valid JSON: {exc}") from exc
