"""
src/database/models.py
───────────────────────
SQLAlchemy ORM model definitions for the ThakAI warehouse tables.

These models translate directly into PostgreSQL tables on Amazon RDS.
The schema mirrors the table names used in the documentation
(documents_raw, document_pages) and adds the additional tables that
the v2 pipeline requires (document_nodes, document_metadata).

Relationship diagram:
    documents_raw
        │  (one-to-many)
        ├─── document_pages
        │        │  (derived by structural extractor)
        │        └─── document_nodes (chapters, articles, clauses)
        └─── document_metadata

Design notes:
  - All primary keys use server-generated UUIDs (gen_random_uuid()) to
    avoid sequence contention in concurrent batch inserts.
  - Arabic text columns are explicitly typed as TEXT (not VARCHAR) because
    legal documents can be very long and text length is unpredictable.
  - The `extra` JSONB column on several tables allows the pipeline to
    store model-specific supplementary data without schema migrations.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """
    Shared declarative base for all ORM models.
    Using the new SQLAlchemy 2.x style DeclarativeBase.
    """
    pass


class DocumentRaw(Base):
    """
    documents_raw
    ─────────────
    The first persistent table in the pipeline.  A row is inserted
    here as soon as the pre-processing stage completes for a document.

    This table is the authoritative record of what was originally
    extracted from S3 — downstream tables reference it but never modify it.
    """
    __tablename__ = "documents_raw"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        comment="Stable document identifier — same UUID across all pipeline runs.",
    )
    # S3 source reference
    file_path = Column(String(1024), nullable=False, unique=True,
                       comment="Full S3 key, e.g. Laws/arabic/decree_01_2024.pdf")
    s3_etag = Column(String(64), nullable=True,
                     comment="S3 ETag used to detect re-uploads / content changes.")

    # Raw document text
    title = Column(Text, nullable=True, comment="Document title extracted from first page or metadata.")
    text_full = Column(Text, nullable=False,
                       comment="Original extracted text, lightly sanitised only.")
    text_clean = Column(Text, nullable=False,
                        comment="Fully normalised Arabic text; primary input for v2 pipeline.")

    # Classification
    jurisdiction = Column(String(64), nullable=True, comment="e.g. UAE, Dubai, Abu Dhabi")
    language = Column(String(32), nullable=False, default="arabic")
    entity = Column(String(256), nullable=True, comment="Issuing government entity.")

    # Pipeline tracking
    ingestion_status = Column(
        String(32), nullable=False, default="ingested",
        comment="Values: ingested | profiled | structured | metadata_extracted | complete | error"
    )
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(),
                        onupdate=func.now())

    # Relationships
    pages = relationship("DocumentPage", back_populates="document",
                         cascade="all, delete-orphan")
    metadata_record = relationship("DocumentMetadata", back_populates="document",
                                   uselist=False, cascade="all, delete-orphan")


class DocumentPage(Base):
    """
    document_pages
    ───────────────
    Created by Stage 1 (Page Splitting).  Each row represents a single
    logical page of the document, aligned with the original PDF layout.
    """
    __tablename__ = "document_pages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents_raw.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number = Column(Integer, nullable=False,
                         comment="1-indexed page number within the document.")
    text_content = Column(Text, nullable=False)
    page_boundary_type = Column(
        String(32), nullable=False, default="real",
        comment="'real' = aligns with PDF page; 'logical' = inferred segment."
    )
    strategy_used = Column(
        String(64), nullable=True,
        comment="Splitting strategy that produced this page, e.g. 'marker_detection'."
    )
    corresponds_to_pdf_page = Column(Boolean, nullable=False, default=True)
    extra = Column(JSONB, nullable=True, comment="Additional page-level metadata.")

    document = relationship("DocumentRaw", back_populates="pages")
    nodes = relationship("DocumentNode", back_populates="page",
                         cascade="all, delete-orphan")


class DocumentNode(Base):
    """
    document_nodes
    ───────────────
    Created by Stage 3 (Structural Extraction).  Represents a single
    hierarchical node in the legal document tree — a chapter, article,
    clause, or sub-clause.

    Parent-child relationships are represented via the `parent_id` self-referential
    foreign key, allowing arbitrary-depth legal hierarchies to be stored in a
    single flat table with efficient recursive CTE queries.
    """
    __tablename__ = "document_nodes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents_raw.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_id = Column(
        UUID(as_uuid=True),
        ForeignKey("document_pages.id", ondelete="SET NULL"),
        nullable=True,
    )
    parent_id = Column(
        UUID(as_uuid=True),
        ForeignKey("document_nodes.id", ondelete="CASCADE"),
        nullable=True,
        comment="NULL for top-level nodes (chapters).",
    )

    # Node classification
    node_type = Column(
        String(32), nullable=False,
        comment="e.g. chapter (باب), article (مادة), clause (فقرة), preamble (ديباجة)"
    )
    node_type_arabic = Column(
        String(64), nullable=True,
        comment="The exact Arabic term used in the document, e.g. 'الباب', 'المادة'."
    )
    node_number = Column(String(32), nullable=True,
                         comment="Number as it appears in the document, e.g. '1', 'الأول'.")
    depth = Column(Integer, nullable=False, default=0,
                   comment="0=chapter, 1=article, 2=clause, 3=sub-clause")

    text_content = Column(Text, nullable=False,
                          comment="Full text of this node including its heading.")
    heading = Column(Text, nullable=True,
                     comment="Heading line only (e.g. 'الباب الأول: أحكام عامة').")

    sequence_index = Column(
        Integer, nullable=False, default=0,
        comment="Zero-based order of this node among its siblings."
    )

    extra = Column(JSONB, nullable=True)

    page = relationship("DocumentPage", back_populates="nodes")
    children = relationship(
        "DocumentNode",
        backref="parent",
        foreign_keys=[parent_id],
        cascade="all, delete-orphan",
    )


class DocumentMetadata(Base):
    """
    document_metadata
    ─────────────────
    Created by Stage 4 (Metadata Extraction).  Stores the structured
    legal metadata extracted by the LLM — used for indexing, filtering,
    and retrieval in downstream applications.
    """
    __tablename__ = "document_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents_raw.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    # Core legal metadata
    title = Column(Text, nullable=True, comment="Full document title in Arabic.")
    title_en = Column(Text, nullable=True, comment="English translation of the title, if available.")
    law_number = Column(String(64), nullable=True, comment="e.g. 'Federal Law No. 1 of 2024'")
    document_type = Column(
        String(64), nullable=True,
        comment="e.g. Law (قانون), Decree (مرسوم), Resolution (قرار), Regulation (لائحة)"
    )
    issuing_authority = Column(Text, nullable=True)
    jurisdiction = Column(String(64), nullable=True, comment="e.g. UAE, Dubai, Abu Dhabi")
    language = Column(String(32), nullable=False, default="arabic")

    # Temporal metadata
    issue_date = Column(String(32), nullable=True,
                        comment="Date as it appears in the document text (Arabic or ISO).")
    effective_date = Column(String(32), nullable=True)
    publication_date = Column(String(32), nullable=True)

    # Status
    status = Column(
        String(32), nullable=True,
        comment="e.g. in_force, repealed, amended, draft"
    )
    amends_document = Column(String(256), nullable=True,
                              comment="Reference to the law this document amends, if applicable.")

    # Structural profile (from Stage 2)
    structural_vocabulary = Column(
        JSONB, nullable=True,
        comment="JSON array of detected division types with their regex patterns."
    )
    hierarchy = Column(
        JSONB, nullable=True,
        comment="Ordered list of hierarchy levels, e.g. ['باب', 'مادة', 'فقرة']."
    )

    # Quality metrics
    extraction_confidence = Column(Float, nullable=True,
                                   comment="0-1 confidence score reported by the LLM.")
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("DocumentRaw", back_populates="metadata_record")
