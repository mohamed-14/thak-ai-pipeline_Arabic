"""
src/ingestion/s3_retriever.py
──────────────────────────────
Handles downloading legal PDF documents from the ThakAI Amazon S3 bucket.

The bucket follows this key structure:
    thakai-documents/
        Laws/
            english/
            arabic/
        Regulatory/
            english/
            arabic/

This module provides both single-document retrieval and batch listing
utilities.  All S3 interaction goes through boto3; credentials are
resolved via the standard AWS credential chain (env vars → ~/.aws →
IAM instance profile), so no credentials are ever hard-coded here.
"""

import io
import logging
from pathlib import Path
from typing import Generator

import boto3
from botocore.exceptions import ClientError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from config.settings import settings

logger = logging.getLogger(__name__)


class S3Retriever:
    """
    Thin wrapper around the boto3 S3 client that adds retry logic,
    structured error handling, and convenience iterators for the
    ThakAI bucket layout.
    """

    def __init__(self) -> None:
        # boto3 resolves credentials automatically from the environment /
        # IAM role — no explicit key passing needed in production.
        self._client = boto3.client(
            "s3",
            region_name=settings.aws_region,
            # Only pass explicit keys when they are provided (useful for local dev).
            **(
                {
                    "aws_access_key_id": settings.aws_access_key_id,
                    "aws_secret_access_key": settings.aws_secret_access_key,
                }
                if settings.aws_access_key_id
                else {}
            ),
        )
        self.bucket = settings.s3_bucket_name

    @retry(
        retry=retry_if_exception_type(ClientError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def download_to_bytes(self, s3_key: str) -> bytes:
        """
        Download a single S3 object and return its content as bytes.

        The @retry decorator handles transient network / throttling errors
        with exponential back-off — important when running batch jobs
        that hit S3 at high concurrency.

        Parameters
        ----------
        s3_key : str
            Full S3 object key, e.g. "Laws/arabic/federal_law_01_2024.pdf"

        Returns
        -------
        bytes
            Raw PDF bytes ready for extraction.
        """
        logger.info("Downloading s3://%s/%s", self.bucket, s3_key)
        buffer = io.BytesIO()
        self._client.download_fileobj(self.bucket, s3_key, buffer)
        buffer.seek(0)
        return buffer.read()

    def download_to_file(self, s3_key: str, local_path: Path) -> Path:
        """
        Download an S3 object to a local file path.
        Creates parent directories if they don't exist.
        Useful for debugging and local development runs.
        """
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading s3://%s/%s → %s", self.bucket, s3_key, local_path)
        self._client.download_file(self.bucket, s3_key, str(local_path))
        return local_path

    def list_documents(
        self,
        language: str = "arabic",
        doc_type: str = "Laws",
    ) -> Generator[str, None, None]:
        """
        Yield all S3 keys for PDF documents under a given language / type prefix.

        Parameters
        ----------
        language : str
            "arabic" or "english"
        doc_type : str
            "Laws" or "Regulatory"

        Yields
        ------
        str
            S3 key for each PDF object found under the prefix.
        """
        prefix = f"{doc_type}/{language}/"
        paginator = self._client.get_paginator("list_objects_v2")

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key: str = obj["Key"]
                if key.lower().endswith(".pdf"):
                    logger.debug("Found document: %s", key)
                    yield key

    def object_exists(self, s3_key: str) -> bool:
        """Check whether an object exists in the bucket without downloading it."""
        try:
            self._client.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "404":
                return False
            raise

    def get_object_metadata(self, s3_key: str) -> dict:
        """Return the S3 object's metadata dict (ETag, ContentLength, etc.)."""
        response = self._client.head_object(Bucket=self.bucket, Key=s3_key)
        return {
            "etag": response.get("ETag", "").strip('"'),
            "size_bytes": response.get("ContentLength", 0),
            "last_modified": response.get("LastModified"),
            "content_type": response.get("ContentType", ""),
        }
