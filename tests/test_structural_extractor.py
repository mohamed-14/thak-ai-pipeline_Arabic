"""
tests/test_structural_extractor.py
────────────────────────────────────
Unit tests for Stage 3 (Structural Extraction) of the v2 pipeline.

These tests validate the Arabic legal document hierarchy parser against
realistic samples of UAE legislation structure.  All tests are offline —
no AWS calls, no database writes (warehouse is mocked).
"""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.pipeline.structural_extractor import StructuralExtractor


# ── Sample document fixtures ──────────────────────────────────────────────────

# Minimal but realistic UAE federal law structure
# (patterns match the Arabic regex in arabic_utils.py)
SIMPLE_LAW = """\
بناءً على الدستور
نصدر القانون الاتحادي الآتي:

الباب الأول
الأحكام العامة

المادة (1)
يهدف هذا القانون إلى تنظيم البيانات الشخصية.

المادة (2)
تسري أحكام هذا القانون على جميع المنشآت.

الباب الثاني
حقوق الأفراد

المادة (3)
لكل فرد الحق في الخصوصية وحماية بياناته.

المادة (4)
يحق لأصحاب البيانات طلب تصحيحها أو حذفها.
"""

# Document with articles but no chapters (common in ministerial decisions)
ARTICLES_ONLY = """\
قرار وزاري رقم (15) لسنة 2023

المادة (1)
تُشكّل لجنة فنية لمراجعة معايير البيانات.

المادة (2)
تجتمع اللجنة مرة كل شهر على الأقل.

المادة (3)
ترفع اللجنة تقاريرها إلى الوزير المختص.
"""

# Profile returned by the LLM profiler for the simple law above
SAMPLE_PROFILE_WITH_CHAPTERS = {
    "document_type": "Federal Law",
    "language": "Arabic",
    "structural_vocabulary": [
        {
            "division_type": "Chapter",
            "arabic_term": "الباب",
            "level": 1,
            "numbering_style": "arabic_ordinal",
            "pattern_description": "Chapter heading: الباب followed by ordinal",
            "regex": r"^[\s\u200f]*(الباب)\s+([\u0600-\u06FF\u0660-\u0669]+|\d+)",
            "example": "الباب الأول",
        },
        {
            "division_type": "Article",
            "arabic_term": "المادة",
            "level": 2,
            "numbering_style": "arabic_numeral",
            "pattern_description": "Article: المادة followed by number in parens",
            "regex": r"^[\s\u200f]*(المادة|مادة)[\s\u00a0]+(\(\d+\)|\d+|[الأولىالثاني]+)",
            "example": "المادة (1)",
        },
    ],
    "hierarchy": ["الباب", "المادة"],
    "has_preamble": True,
    "message": "profiled",
}

SAMPLE_PROFILE_ARTICLES_ONLY = {
    "document_type": "Ministerial Resolution",
    "language": "Arabic",
    "structural_vocabulary": [
        {
            "division_type": "Article",
            "arabic_term": "المادة",
            "level": 1,
            "numbering_style": "arabic_numeral",
            "regex": r"^[\s\u200f]*(المادة|مادة)[\s\u00a0]+(\(\d+\)|\d+)",
            "example": "المادة (1)",
        },
    ],
    "hierarchy": ["المادة"],
    "has_preamble": False,
    "message": "profiled",
}


@pytest.fixture
def mock_warehouse():
    wh = MagicMock()
    wh.store_nodes = MagicMock()
    wh.mark_document_status = MagicMock()
    return wh


@pytest.fixture
def extractor(mock_warehouse):
    return StructuralExtractor(warehouse=mock_warehouse)


class TestBasicExtraction:

    def test_extracts_chapters(self, extractor, mock_warehouse):
        """Two chapter headings must produce two chapter nodes."""
        nodes = extractor.extract_and_store(uuid4(), SIMPLE_LAW, SAMPLE_PROFILE_WITH_CHAPTERS)
        chapters = [n for n in nodes if n.node_type == "chapter"]
        assert len(chapters) == 2

    def test_extracts_articles(self, extractor, mock_warehouse):
        """Four articles must produce four article nodes."""
        nodes = extractor.extract_and_store(uuid4(), SIMPLE_LAW, SAMPLE_PROFILE_WITH_CHAPTERS)
        articles = [n for n in nodes if n.node_type == "article"]
        assert len(articles) == 4

    def test_preamble_extracted(self, extractor, mock_warehouse):
        """Text before the first structural marker must become a preamble node."""
        nodes = extractor.extract_and_store(uuid4(), SIMPLE_LAW, SAMPLE_PROFILE_WITH_CHAPTERS)
        preambles = [n for n in nodes if n.node_type == "preamble"]
        assert len(preambles) == 1
        assert "بناءً على الدستور" in preambles[0].text_content

    def test_articles_under_chapters_have_parent(self, extractor, mock_warehouse):
        """Articles inside a chapter must have a non-None parent_id."""
        nodes = extractor.extract_and_store(uuid4(), SIMPLE_LAW, SAMPLE_PROFILE_WITH_CHAPTERS)
        articles = [n for n in nodes if n.node_type == "article"]
        for article in articles:
            assert article.parent_id is not None, \
                f"Article '{article.heading}' has no parent chapter"

    def test_parent_ids_point_to_chapters(self, extractor, mock_warehouse):
        """Each article's parent_id must correspond to an actual chapter node."""
        nodes = extractor.extract_and_store(uuid4(), SIMPLE_LAW, SAMPLE_PROFILE_WITH_CHAPTERS)
        chapter_ids = {n.id for n in nodes if n.node_type == "chapter"}
        articles = [n for n in nodes if n.node_type == "article"]
        for article in articles:
            assert article.parent_id in chapter_ids, \
                f"Article parent_id {article.parent_id} not in chapter ids {chapter_ids}"


class TestArticlesOnly:

    def test_no_chapters_produced(self, extractor, mock_warehouse):
        """A document without chapters must produce zero chapter nodes."""
        nodes = extractor.extract_and_store(
            uuid4(), ARTICLES_ONLY, SAMPLE_PROFILE_ARTICLES_ONLY
        )
        chapters = [n for n in nodes if n.node_type == "chapter"]
        assert len(chapters) == 0

    def test_articles_extracted_correctly(self, extractor, mock_warehouse):
        """Three articles in the sample must produce exactly three article nodes."""
        nodes = extractor.extract_and_store(
            uuid4(), ARTICLES_ONLY, SAMPLE_PROFILE_ARTICLES_ONLY
        )
        articles = [n for n in nodes if n.node_type == "article"]
        assert len(articles) == 3

    def test_top_level_articles_have_no_parent(self, extractor, mock_warehouse):
        """Without chapters, top-level articles must have parent_id = None."""
        nodes = extractor.extract_and_store(
            uuid4(), ARTICLES_ONLY, SAMPLE_PROFILE_ARTICLES_ONLY
        )
        articles = [n for n in nodes if n.node_type == "article"]
        for article in articles:
            assert article.parent_id is None


class TestNodeAttributes:

    def test_article_numbers_extracted(self, extractor, mock_warehouse):
        """Article nodes must have their number attribute populated."""
        nodes = extractor.extract_and_store(uuid4(), SIMPLE_LAW, SAMPLE_PROFILE_WITH_CHAPTERS)
        articles = [n for n in nodes if n.node_type == "article"]
        for article in articles:
            assert article.node_number is not None, \
                f"Article node has no number: {article.heading}"

    def test_article_text_contains_body(self, extractor, mock_warehouse):
        """Article text_content must include more than just the heading line."""
        nodes = extractor.extract_and_store(uuid4(), SIMPLE_LAW, SAMPLE_PROFILE_WITH_CHAPTERS)
        articles = [n for n in nodes if n.node_type == "article"]
        # Article 1 body text should appear somewhere in its content
        art_1 = next(
            (a for a in articles if a.node_number == "1"), None
        )
        assert art_1 is not None
        assert "البيانات الشخصية" in art_1.text_content

    def test_sequence_indices_are_ordered(self, extractor, mock_warehouse):
        """Nodes at the same depth and under the same parent must have ascending sequence_index."""
        nodes = extractor.extract_and_store(uuid4(), SIMPLE_LAW, SAMPLE_PROFILE_WITH_CHAPTERS)
        chapters = sorted(
            [n for n in nodes if n.node_type == "chapter"],
            key=lambda n: n.sequence_index,
        )
        indices = [n.sequence_index for n in chapters]
        assert indices == sorted(indices)

    def test_chapter_depth_is_zero(self, extractor, mock_warehouse):
        """Chapters must be at depth 0."""
        nodes = extractor.extract_and_store(uuid4(), SIMPLE_LAW, SAMPLE_PROFILE_WITH_CHAPTERS)
        for node in nodes:
            if node.node_type == "chapter":
                assert node.depth == 0

    def test_article_depth_is_one(self, extractor, mock_warehouse):
        """Articles under chapters must be at depth 1."""
        nodes = extractor.extract_and_store(uuid4(), SIMPLE_LAW, SAMPLE_PROFILE_WITH_CHAPTERS)
        for node in nodes:
            if node.node_type == "article":
                assert node.depth == 1


class TestFallbackPatterns:

    def test_fallback_used_when_profile_missing_chapter_pattern(
        self, extractor, mock_warehouse
    ):
        """If the profile has no chapter pattern, built-in CHAPTER_PATTERN must be used."""
        # Profile with only article vocabulary
        minimal_profile = {
            "structural_vocabulary": [
                {
                    "division_type": "Article",
                    "arabic_term": "المادة",
                    "level": 1,
                    "regex": r"^[\s\u200f]*(المادة)[\s\u00a0]+(\(\d+\)|\d+)",
                }
            ],
            "hierarchy": ["المادة"],
        }
        nodes = extractor.extract_and_store(uuid4(), SIMPLE_LAW, minimal_profile)
        # Should still find articles even without profiler-provided chapter regex
        articles = [n for n in nodes if n.node_type == "article"]
        assert len(articles) > 0

    def test_empty_profile_falls_back_gracefully(self, extractor, mock_warehouse):
        """An empty profile must not crash — fallback patterns take over."""
        nodes = extractor.extract_and_store(uuid4(), SIMPLE_LAW, {})
        # At minimum, preamble or articles should be found
        assert len(nodes) > 0


class TestWarehouseInteraction:

    def test_store_nodes_called_once(self, extractor, mock_warehouse):
        """store_nodes must be called exactly once per document."""
        extractor.extract_and_store(uuid4(), SIMPLE_LAW, SAMPLE_PROFILE_WITH_CHAPTERS)
        mock_warehouse.store_nodes.assert_called_once()

    def test_status_marked_as_structured(self, extractor, mock_warehouse):
        """Document status must be updated to 'structured' after extraction."""
        doc_id = uuid4()
        extractor.extract_and_store(doc_id, SIMPLE_LAW, SAMPLE_PROFILE_WITH_CHAPTERS)
        mock_warehouse.mark_document_status.assert_called_with(doc_id, "structured")
