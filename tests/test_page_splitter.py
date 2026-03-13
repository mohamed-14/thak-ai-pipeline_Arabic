"""
tests/test_page_splitter.py
────────────────────────────
Unit tests for Stage 1 (Page Splitting) of the v2 pipeline.

Uses pytest-mock to replace the WarehouseClient with a mock so these
tests have zero database dependency.
"""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.pipeline.page_splitter import PageSplitter, _CHARS_PER_LOGICAL_PAGE


@pytest.fixture
def mock_warehouse():
    """Return a MagicMock that replaces WarehouseClient."""
    wh = MagicMock()
    wh.store_pages = MagicMock()
    return wh


@pytest.fixture
def splitter(mock_warehouse):
    return PageSplitter(warehouse=mock_warehouse)


class TestMarkerDetection:
    """Tests for the form-feed marker-based splitting strategy."""

    def test_splits_on_form_feed(self, splitter, mock_warehouse):
        """Form-feeds must produce exactly one page per segment."""
        text = "صفحة أولى\nبعض النص\f\nصفحة ثانية\nمزيد\f\nصفحة ثالثة"
        doc_id = uuid4()
        pages = splitter.split_and_store(doc_id, text)

        assert len(pages) == 3
        assert pages[0].page_number == 1
        assert pages[1].page_number == 2
        assert pages[2].page_number == 3

    def test_page_boundary_type_is_real(self, splitter, mock_warehouse):
        """Pages from form-feed splitting must be marked as real boundaries."""
        text = "صفحة أولى\f\nصفحة ثانية"
        pages = splitter.split_and_store(uuid4(), text)

        for page in pages:
            assert page.page_boundary_type == "real"
            assert page.corresponds_to_pdf_page is True

    def test_strategy_recorded(self, splitter, mock_warehouse):
        text = "صفحة أولى\f\nصفحة ثانية"
        pages = splitter.split_and_store(uuid4(), text)
        for page in pages:
            assert page.strategy_used == "marker_detection"

    def test_blank_pages_excluded_from_db(self, splitter, mock_warehouse):
        """Pages that contain only whitespace must not be written to the DB."""
        text = "صفحة مفيدة\f\n   \n\f\nصفحة مفيدة أخرى"
        splitter.split_and_store(uuid4(), text)
        # Inspect what was passed to store_pages
        call_args = mock_warehouse.store_pages.call_args
        stored_pages = call_args[0][1]  # second positional arg
        for page_dict in stored_pages:
            assert page_dict["text_content"].strip() != ""

    def test_single_page_document(self, splitter, mock_warehouse):
        """A document with no form-feeds still produces one page."""
        text = "المادة الأولى: أحكام عامة"
        pages = splitter.split_and_store(uuid4(), text)
        # Without \f, falls back to length_heuristic — but with short text, 1 page
        assert len(pages) == 1

    def test_store_pages_called_once(self, splitter, mock_warehouse):
        """store_pages must be called exactly once per document."""
        splitter.split_and_store(uuid4(), "نص\f\nنص آخر")
        mock_warehouse.store_pages.assert_called_once()


class TestLengthHeuristic:
    """Tests for the fallback length-based splitting strategy."""

    def test_fallback_used_without_formfeed(self, splitter, mock_warehouse):
        """When no form-feed is present, length_heuristic strategy is used."""
        # Build text longer than one logical page
        long_para = "هذه فقرة طويلة نسبياً. " * 120
        text = long_para
        pages = splitter.split_and_store(uuid4(), text)

        for page in pages:
            assert page.strategy_used == "length_heuristic"
            assert page.page_boundary_type == "logical"
            assert page.corresponds_to_pdf_page is False

    def test_long_document_produces_multiple_pages(self, splitter, mock_warehouse):
        """A text much longer than _CHARS_PER_LOGICAL_PAGE must produce > 1 page."""
        # Paragraphs separated by double-newlines (what _split_by_length splits on).
        # Each paragraph is ~90 chars; we need enough to exceed _CHARS_PER_LOGICAL_PAGE.
        para = "هذا نص طويل جداً في القانون الاتحادي تتضمن معلومات تفصيلية حول الأحكام. "
        paras_needed = (_CHARS_PER_LOGICAL_PAGE // len(para)) + 5
        text = "\n\n".join([para] * paras_needed)
        pages = splitter.split_and_store(uuid4(), text)
        assert len(pages) > 1

    def test_page_numbers_are_sequential(self, splitter, mock_warehouse):
        """Page numbers must start at 1 and increment by 1."""
        long_text = ("فقرة طويلة من النص القانوني. " * 200)
        pages = splitter.split_and_store(uuid4(), long_text)
        page_numbers = [p.page_number for p in pages]
        assert page_numbers == list(range(1, len(page_numbers) + 1))

    def test_short_document_stays_one_page(self, splitter, mock_warehouse):
        """A short document (under one page budget) must remain a single page."""
        short_text = "المادة الأولى: أهداف القانون\nالمادة الثانية: التعريفات"
        pages = splitter.split_and_store(uuid4(), short_text)
        assert len(pages) == 1


class TestDocumentId:
    """Tests that document_id is correctly forwarded to the warehouse."""

    def test_correct_document_id_stored(self, splitter, mock_warehouse):
        doc_id = uuid4()
        splitter.split_and_store(doc_id, "نص\f\nنص")
        stored_doc_id = mock_warehouse.store_pages.call_args[0][0]
        assert stored_doc_id == doc_id
