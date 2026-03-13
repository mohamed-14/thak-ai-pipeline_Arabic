"""
src/ingestion/ocr_fallback.py
──────────────────────────────
Amazon Textract-based OCR fallback for scanned Arabic PDFs.

Why Textract (not open-source OCR)?
  - Aligns with the ThakAI AWS-native constraint.
  - Textract has native Arabic support with right-to-left reading order
    correction, which tools like Tesseract struggle with on dense legal PDFs.
  - The async Textract API handles multi-page PDFs natively when the PDF
    is stored in S3, which is our exact use case.
  - Confidence scores per word/block let us filter out low-quality reads.

Two modes are supported:
  1. Synchronous (detect_document_text) — for single pages / small images.
     We use this for individual pages flagged by the PDF extractor.
  2. Asynchronous (start_document_text_detection) — for full multi-page PDFs
     where most pages need OCR.  This is the cost-efficient path for entirely
     scanned documents.
"""

import logging
import time
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from config.settings import settings

logger = logging.getLogger(__name__)


class TextractOCR:
    """
    Wraps the Amazon Textract API to extract Arabic text from page images
    or S3-hosted PDFs.

    The class exposes two methods:
      - `ocr_page_image`  : synchronous, for JPEG bytes of a single page.
      - `ocr_s3_document` : asynchronous, for a full PDF already in S3.
    """

    # Textract's synchronous API accepts images up to 5 MB.
    _MAX_SYNC_BYTES = 5 * 1024 * 1024

    def __init__(self) -> None:
        self._client = boto3.client(
            "textract",
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
        self.confidence_threshold = settings.textract_confidence_threshold

    def ocr_page_image(self, image_bytes: bytes) -> str:
        """
        Run synchronous Textract OCR on a single page image (JPEG bytes).

        Textract returns a list of Block objects.  We filter to LINE-type blocks
        with confidence above the configured threshold, then join them with
        newlines preserving reading order (Textract returns lines top-to-bottom,
        right-to-left for Arabic, which aligns with PDF page order).

        Parameters
        ----------
        image_bytes : bytes
            JPEG-encoded page image, as produced by PDFExtractor._rasterise_page.

        Returns
        -------
        str
            Extracted text lines joined by newlines.
        """
        if len(image_bytes) > self._MAX_SYNC_BYTES:
            logger.warning(
                "Image size (%d bytes) exceeds Textract sync limit; "
                "quality may be reduced after JPEG recompression.",
                len(image_bytes),
            )

        try:
            response = self._client.detect_document_text(
                Document={"Bytes": image_bytes}
            )
        except ClientError as exc:
            logger.error("Textract sync OCR failed: %s", exc)
            raise

        lines: list[str] = []
        for block in response.get("Blocks", []):
            if (
                block["BlockType"] == "LINE"
                and block.get("Confidence", 0) >= self.confidence_threshold
            ):
                lines.append(block["Text"])

        logger.debug("Textract extracted %d lines from page image.", len(lines))
        return "\n".join(lines)

    def ocr_s3_document(
        self,
        s3_key: str,
        bucket: Optional[str] = None,
        poll_interval_seconds: int = 5,
        max_wait_seconds: int = 300,
    ) -> dict[int, str]:
        """
        Run asynchronous Textract OCR on a full PDF stored in S3.

        This is the preferred path for entirely scanned documents because:
          - A single API call handles all pages.
          - Textract's async pipeline is more accurate on multi-page Arabic
            docs than running per-page sync calls independently.
          - No per-page image rasterisation is needed.

        Returns
        -------
        dict[int, str]
            Mapping of {page_number (1-indexed): extracted_text}.
            Pages that returned no confident text are omitted.
        """
        bucket = bucket or settings.s3_bucket_name
        logger.info("Starting async Textract job for s3://%s/%s", bucket, s3_key)

        # Start the async job
        start_response = self._client.start_document_text_detection(
            DocumentLocation={"S3Object": {"Bucket": bucket, "Name": s3_key}}
        )
        job_id: str = start_response["JobId"]
        logger.info("Textract job started: %s", job_id)

        # Poll until complete
        elapsed = 0
        while elapsed < max_wait_seconds:
            time.sleep(poll_interval_seconds)
            elapsed += poll_interval_seconds

            result = self._client.get_document_text_detection(JobId=job_id)
            status = result["JobStatus"]

            if status == "SUCCEEDED":
                logger.info("Textract job %s completed in %ds.", job_id, elapsed)
                return self._parse_async_result(job_id)
            elif status == "FAILED":
                reason = result.get("StatusMessage", "unknown reason")
                raise RuntimeError(f"Textract job {job_id} failed: {reason}")
            else:
                logger.debug(
                    "Textract job %s status: %s (%ds elapsed)", job_id, status, elapsed
                )

        raise TimeoutError(
            f"Textract job {job_id} did not complete within {max_wait_seconds}s."
        )

    def _parse_async_result(self, job_id: str) -> dict[int, str]:
        """
        Paginate through all Textract result pages for an async job and
        assemble a page-number → text mapping.
        """
        pages: dict[int, list[str]] = {}
        next_token: Optional[str] = None

        while True:
            kwargs: dict = {"JobId": job_id}
            if next_token:
                kwargs["NextToken"] = next_token

            response = self._client.get_document_text_detection(**kwargs)

            for block in response.get("Blocks", []):
                if (
                    block["BlockType"] == "LINE"
                    and block.get("Confidence", 0) >= self.confidence_threshold
                ):
                    page_num: int = block.get("Page", 1)
                    pages.setdefault(page_num, []).append(block["Text"])

            next_token = response.get("NextToken")
            if not next_token:
                break

        return {page_num: "\n".join(lines) for page_num, lines in pages.items()}
