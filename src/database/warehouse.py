"""
src/database/warehouse.py
──────────────────────────
Database session management and CRUD helpers for the ThakAI warehouse.

All pipeline stages interact with PostgreSQL exclusively through this
module.  No stage should ever import SQLAlchemy directly — this keeps
the persistence layer cleanly separated from business logic.

The module provides:
  1. A `get_session()` context manager for safe transaction handling.
  2. `WarehouseClient` — a higher-level API with named methods that
     map to the pipeline's storage requirements.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional
from uuid import UUID

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import settings
from src.database.models import (
    Base,
    DocumentMetadata,
    DocumentNode,
    DocumentPage,
    DocumentRaw,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine + Session factory
# ---------------------------------------------------------------------------

# `pool_pre_ping=True` instructs SQLAlchemy to issue a lightweight "SELECT 1"
# before handing a connection from the pool.  This prevents stale-connection
# errors that are common with long-running batch jobs on RDS.
_engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    echo=False,  # Set True to log all SQL — useful for debugging, noisy in prod.
)

_SessionFactory = sessionmaker(bind=_engine, autoflush=False, autocommit=False)


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Provide a transactional database session.

    Usage:
        with get_session() as session:
            session.add(some_orm_object)

    The context manager commits on clean exit and rolls back on any
    exception, then closes the session to return the connection to the pool.
    This pattern is safer than relying on callers to remember commit/rollback.
    """
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """
    Create all tables defined in models.py if they do not already exist.
    Safe to call on every startup — SQLAlchemy uses CREATE TABLE IF NOT EXISTS.
    In production, prefer Alembic migrations over this function.
    """
    logger.info("Initialising database schema on %s ...", settings.db_host)
    Base.metadata.create_all(_engine)
    logger.info("Database schema ready.")


# ---------------------------------------------------------------------------
# WarehouseClient
# ---------------------------------------------------------------------------

class WarehouseClient:
    """
    High-level persistence API consumed by the pipeline stages.

    Rather than having each stage manage its own SQL, all writes and
    targeted reads go through this client, which keeps the stage code
    focused on business logic.
    """

    # ── Pre-Processing Stage ──────────────────────────────────────────────────

    def upsert_document_raw(
        self,
        file_path: str,
        text_full: str,
        text_clean: str,
        language: str = "arabic",
        s3_etag: Optional[str] = None,
        title: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        entity: Optional[str] = None,
    ) -> DocumentRaw:
        """
        Insert a new document_raw record, or update it if a record with the
        same file_path already exists (e.g. re-processing an updated PDF).

        Returns the persisted ORM object with its UUID populated.
        """
        with get_session() as session:
            doc = session.query(DocumentRaw).filter_by(file_path=file_path).first()

            if doc is None:
                doc = DocumentRaw(
                    file_path=file_path,
                    text_full=text_full,
                    text_clean=text_clean,
                    language=language,
                    s3_etag=s3_etag,
                    title=title,
                    jurisdiction=jurisdiction,
                    entity=entity,
                    ingestion_status="ingested",
                )
                session.add(doc)
                logger.info("Inserted new document_raw for '%s'", file_path)
            else:
                # Re-process: update text fields and reset status
                doc.text_full = text_full
                doc.text_clean = text_clean
                doc.s3_etag = s3_etag or doc.s3_etag
                doc.ingestion_status = "ingested"
                doc.error_message = None
                logger.info("Updated existing document_raw for '%s'", file_path)

            session.flush()
            # Expunge so the object can be used outside the session
            session.expunge(doc)
            return doc

    def mark_document_status(
        self,
        document_id: UUID,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Update the ingestion_status (and optionally error_message) of a document."""
        with get_session() as session:
            doc = session.get(DocumentRaw, document_id)
            if doc:
                doc.ingestion_status = status
                doc.error_message = error_message

    # ── Stage 1: Page Splitting ───────────────────────────────────────────────

    def store_pages(self, document_id: UUID, pages: list[dict]) -> None:
        """
        Bulk-insert document_pages rows for a document, replacing any
        existing pages (to support idempotent re-processing).

        Parameters
        ----------
        document_id : UUID
        pages : list[dict]
            Each dict should contain: page_number, text_content,
            page_boundary_type, strategy_used, corresponds_to_pdf_page,
            and optionally extra.
        """
        with get_session() as session:
            # Delete existing pages for this document before re-inserting
            session.query(DocumentPage).filter_by(document_id=document_id).delete()

            for page_data in pages:
                page = DocumentPage(document_id=document_id, **page_data)
                session.add(page)

            logger.info(
                "Stored %d pages for document %s", len(pages), document_id
            )

    # ── Stage 3: Structural Extraction ───────────────────────────────────────

    def store_nodes(self, document_id: UUID, nodes: list[dict]) -> None:
        """
        Bulk-insert document_nodes rows for a document.
        Replaces existing nodes on re-processing.
        """
        with get_session() as session:
            session.query(DocumentNode).filter_by(document_id=document_id).delete()
            for node_data in nodes:
                node = DocumentNode(document_id=document_id, **node_data)
                session.add(node)
            logger.info(
                "Stored %d structural nodes for document %s",
                len(nodes),
                document_id,
            )

    # ── Stage 4: Metadata Extraction ─────────────────────────────────────────

    def store_metadata(self, document_id: UUID, metadata: dict) -> None:
        """
        Upsert document_metadata for a document.
        The metadata dict keys should match DocumentMetadata column names.
        """
        with get_session() as session:
            existing = (
                session.query(DocumentMetadata)
                .filter_by(document_id=document_id)
                .first()
            )
            if existing:
                for key, value in metadata.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
            else:
                meta = DocumentMetadata(document_id=document_id, **metadata)
                session.add(meta)
            logger.info("Stored metadata for document %s", document_id)

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_document_by_path(self, file_path: str) -> Optional[DocumentRaw]:
        """Retrieve a document_raw record by its S3 file path."""
        with get_session() as session:
            doc = session.query(DocumentRaw).filter_by(file_path=file_path).first()
            if doc:
                session.expunge(doc)
            return doc

    def list_pending_documents(self, status: str = "ingested") -> list[DocumentRaw]:
        """
        Return all documents with the given ingestion_status.
        Used by the pipeline orchestrator to find work items.
        """
        with get_session() as session:
            docs = (
                session.query(DocumentRaw)
                .filter_by(ingestion_status=status)
                .all()
            )
            for doc in docs:
                session.expunge(doc)
            return docs
