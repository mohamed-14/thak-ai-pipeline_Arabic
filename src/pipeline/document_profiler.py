"""
src/pipeline/document_profiler.py
──────────────────────────────────
Stage 2 of the v2 Pipeline: Document Profiling.

This stage answers the question: "What is the structural vocabulary of
this specific legal document?"

Different UAE legal instruments use different Arabic terms for their
hierarchy levels.  Federal laws (قانون اتحادي) typically use:
  الباب (Chapter) → المادة (Article) → الفقرة/البند (Clause)

Cabinet decisions (قرار مجلس الوزراء) may use:
  المادة → البند → الفقرة

Ministerial resolutions and regulatory documents often have no chapters
at all and go straight from preamble to articles.

Rather than hard-coding a single vocabulary, we send the first N pages
to an AWS Bedrock LLM and ask it to identify the actual patterns used.
The model returns a JSON profile that drives all subsequent regex-based
structural extraction in Stage 3 — giving us both LLM intelligence and
the speed/reliability of deterministic regex for the bulk of processing.

AWS alignment: uses `boto3` with the Bedrock Runtime API (`InvokeModel`).
"""

import json
import logging
from uuid import UUID

import boto3
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings
from src.database.warehouse import WarehouseClient

logger = logging.getLogger(__name__)

# ── Bedrock profiling prompt ──────────────────────────────────────────────────
# The prompt is carefully engineered to:
#   1. Work in Arabic context while receiving instructions in English (Bedrock
#      Claude models handle cross-lingual instructions reliably).
#   2. Produce strict JSON (no markdown fences) so we can parse it directly.
#   3. Include regex patterns in the output — these are used verbatim by Stage 3.

_PROFILING_SYSTEM_PROMPT = """\
You are an expert in Arabic legal document structure, specifically UAE federal
and emirate-level legislation.  You will be given sample pages from an Arabic
legal document and must analyse its structural vocabulary.

Respond ONLY with a valid JSON object — no markdown, no explanation, no preamble.
The JSON must follow this exact schema:
{
  "document_type": "<string: e.g. Federal Law, Cabinet Decision, Ministerial Resolution, Decree>",
  "language": "Arabic",
  "structural_vocabulary": [
    {
      "division_type": "<English name: e.g. Chapter, Article, Clause>",
      "arabic_term": "<Exact Arabic term as it appears, e.g. الباب, المادة, الفقرة>",
      "level": <integer: 1=top, 2=mid, 3=leaf>,
      "numbering_style": "<roman | arabic_numeral | arabic_ordinal | arabic_letter | mixed>",
      "pattern_description": "<brief human-readable description>",
      "regex": "<Python regex pattern to detect the START of this division>",
      "example": "<literal example from the document text>",
      "estimated_count": <integer or null>
    }
  ],
  "hierarchy": ["<level1_arabic_term>", "<level2_arabic_term>", ...],
  "has_preamble": <boolean>,
  "preamble_marker": "<Arabic phrase that starts the preamble, or null>",
  "closing_marker": "<Arabic phrase that ends the document body, or null>",
  "message": "profiled"
}

Important notes for Arabic legal documents:
- Chapter headings often appear on their own line: الباب الأول, الباب الثاني, etc.
- Article markers always start a new line and use المادة followed by a number.
- Numbers may be Arabic-Indic (١٢٣) or Western (123) or ordinal words (الأول).
- The regex must use re.MULTILINE flag and match from the start of a line (^).
- If a division type is not present, omit it from the array.
"""


_PROFILING_USER_TEMPLATE = """\
Here are the first {n_pages} pages of an Arabic legal document.  Analyse the
structural vocabulary and return the JSON profile.

Document text:
---
{sample_text}
---
"""


class DocumentProfiler:
    """
    Uses AWS Bedrock (Claude) to detect and return the structural vocabulary
    of a legal document.  The profile is stored in document_metadata and
    consumed by Stage 3.
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

    def profile(self, document_id: UUID, text_clean: str) -> dict:
        """
        Analyse the document's structural vocabulary and persist the profile.

        Parameters
        ----------
        document_id : UUID
        text_clean : str
            Full normalised document text.

        Returns
        -------
        dict
            The parsed structural profile (same structure as the JSON schema
            documented in the system prompt above).
        """
        sample_text = self._get_sample(text_clean)
        profile = self._call_bedrock(sample_text)

        # Persist the profile to document_metadata so Stage 3 and Stage 4
        # can read it without re-running the expensive LLM call.
        self._db.store_metadata(
            document_id,
            {
                "structural_vocabulary": profile.get("structural_vocabulary"),
                "hierarchy": profile.get("hierarchy"),
                "document_type": profile.get("document_type"),
                "language": profile.get("language", "Arabic"),
            },
        )
        self._db.mark_document_status(document_id, "profiled")
        logger.info(
            "Document %s profiled: type=%s, hierarchy=%s",
            document_id,
            profile.get("document_type"),
            profile.get("hierarchy"),
        )
        return profile

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_sample(self, text_clean: str) -> str:
        """
        Extract the first N pages (or characters) from the document for the
        profiling prompt.  Sending the full document would exceed token limits
        and is unnecessary — the structural vocabulary is apparent from the
        first few pages.
        """
        pages = text_clean.split("\f")
        sample_pages = pages[: settings.profiling_sample_pages]
        sample = "\f".join(sample_pages)

        # Hard cap at 6,000 characters to stay safely within the prompt budget
        # while leaving room for the system prompt and expected JSON response.
        if len(sample) > 6_000:
            sample = sample[:6_000]

        return sample

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        reraise=True,
    )
    def _call_bedrock(self, sample_text: str) -> dict:
        """
        Invoke the Bedrock model with the profiling prompt.
        Retries up to 3 times with exponential back-off to handle throttling.
        """
        n_pages = sample_text.count("\f") + 1
        user_message = _PROFILING_USER_TEMPLATE.format(
            n_pages=n_pages,
            sample_text=sample_text,
        )

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": settings.bedrock_max_tokens,
            "system": _PROFILING_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user_message}],
        }

        response = self._bedrock.invoke_model(
            modelId=settings.bedrock_model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(payload),
        )

        body = json.loads(response["body"].read())
        raw_text: str = body["content"][0]["text"].strip()

        # Strip any accidental markdown code fences the model might add
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        try:
            profile = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            logger.error(
                "Bedrock returned non-JSON profiling output: %s", raw_text[:500]
            )
            raise ValueError(f"Bedrock profiling response was not valid JSON: {exc}") from exc

        return profile
