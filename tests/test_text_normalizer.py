"""
tests/test_text_normalizer.py
──────────────────────────────
Unit tests for Arabic text normalisation.

These tests do NOT require any AWS credentials or database connections.
They validate the core Arabic text processing logic that everything else
in the pipeline depends on — if these tests fail, structural extraction
and embedding quality will both be degraded.
"""

import pytest

from src.utils.arabic_utils import (
    count_arabic_chars,
    extract_law_reference,
    is_arabic_text,
    normalize_arabic,
    remove_extraction_noise,
)


class TestNormalizeArabic:
    """Tests for the normalize_arabic() function."""

    def test_removes_harakat(self):
        """Diacritics must be stripped in all cases."""
        # فَعَلَ (with fatha) → فعل
        assert normalize_arabic("فَعَلَ") == "فعل"

    def test_removes_tatweel(self):
        """Kashida / tatweel elongation must be stripped."""
        assert normalize_arabic("جـميـل") == "جميل"

    def test_normalises_alef_variants(self):
        """All alef variants must normalise to plain alef (ا)."""
        # أحكام and احكام should become identical after normalisation
        assert normalize_arabic("أحكام") == normalize_arabic("احكام")
        assert normalize_arabic("إجراءات") == normalize_arabic("اجراءات")
        assert normalize_arabic("آلية") == normalize_arabic("الية")

    def test_converts_arabic_indic_digits(self):
        """Arabic-Indic numerals must be converted to Western Arabic."""
        assert normalize_arabic("المادة ٣") == "المادة 3"
        assert normalize_arabic("لسنة ٢٠٢٤") == "لسنة 2024"

    def test_converts_extended_arabic_indic(self):
        """Extended (Farsi/Urdu) Arabic-Indic digits must also be converted."""
        assert normalize_arabic("۲۰۲۴") == "2024"

    def test_applies_nfkc_normalization(self):
        """Arabic presentation forms should be NFKC-collapsed to canonical forms."""
        # Lam-alef ligature (U+FEFB) should normalise to ل + ا
        ligature = "\uFEFB"  # ﻻ — lam-alef presentational form
        result = normalize_arabic(ligature)
        # After NFKC, the ligature becomes the two-character sequence لا
        assert "ل" in result or "ا" in result

    def test_empty_string_returns_empty(self):
        assert normalize_arabic("") == ""

    def test_preserves_line_breaks(self):
        """Line breaks must be preserved so structural patterns still work."""
        text = "المادة الأولى\nيهدف هذا القانون"
        result = normalize_arabic(text)
        assert "\n" in result

    def test_collapses_multiple_spaces(self):
        """Multiple spaces within a line must be collapsed to one."""
        text = "المادة  الأولى   من   القانون"
        result = normalize_arabic(text)
        assert "  " not in result

    def test_aggressive_mode_normalises_teh_marbuta(self):
        """Aggressive mode converts ة → ه."""
        assert normalize_arabic("اللجنة", aggressive=True).endswith("ه")

    def test_non_aggressive_preserves_teh_marbuta(self):
        """Default mode preserves ة."""
        assert normalize_arabic("اللجنة", aggressive=False).endswith("ة")

    def test_real_legal_article_normalises(self):
        """End-to-end test with text representative of a real UAE law article."""
        article = (
            "المادة (٣): يَجِبُ عَلى المُنشآتِ التي تَتَولّى مُعالَجَةَ "
            "البياناتِ الشَّخصِيَّةِ اتِّخاذُ الاحتِياطاتِ اللازِمةِ."
        )
        result = normalize_arabic(article)
        # Harakat gone
        assert "َ" not in result
        assert "ِ" not in result
        # Digit converted
        assert "3" in result
        assert "٣" not in result


class TestRemoveExtractionNoise:
    """Tests for PDF-specific noise removal."""

    def test_removes_standalone_page_numbers(self):
        """Standalone numerals on their own line are page number artefacts."""
        text = "نص القانون\n42\nمزيد من النص"
        result = remove_extraction_noise(text)
        assert "42" not in result

    def test_removes_repeated_short_lines(self):
        """Lines that repeat more than 3 times are running headers/footers."""
        repeated = "الجريدة الرسمية"
        text = "\n".join([repeated] * 5 + ["نص القانون الفعلي"])
        result = remove_extraction_noise(text)
        assert "نص القانون الفعلي" in result
        # The repeated line should appear at most once (or not at all)
        assert result.count(repeated) <= 1

    def test_removes_null_bytes(self):
        """Null bytes are common PDF extraction artefacts."""
        text = "نص\x00القانون"
        result = remove_extraction_noise(text)
        assert "\x00" not in result

    def test_preserves_substantive_content(self):
        """Content that does not match noise patterns must be preserved."""
        text = "المادة الأولى: أهداف القانون\nيهدف هذا القانون إلى حماية البيانات."
        result = remove_extraction_noise(text)
        assert "المادة الأولى" in result
        assert "يهدف هذا القانون" in result


class TestIsArabicText:
    """Tests for Arabic text detection."""

    def test_arabic_text_detected(self):
        assert is_arabic_text("هذا نص عربي كامل") is True

    def test_english_text_not_arabic(self):
        assert is_arabic_text("This is English text only") is False

    def test_mixed_text_mostly_arabic(self):
        assert is_arabic_text("القانون رقم 3 لسنة 2022 UAE") is True

    def test_empty_string(self):
        assert is_arabic_text("") is False

    def test_numbers_only_not_arabic(self):
        assert is_arabic_text("1234567890") is False

    def test_custom_threshold(self):
        """A text that is 20% Arabic should pass a threshold of 0.1 but not 0.5."""
        # 2 Arabic chars out of ~10 non-space = 20%
        text = "abc def ال"
        assert is_arabic_text(text, threshold=0.1) is True
        assert is_arabic_text(text, threshold=0.5) is False


class TestCountArabicChars:
    """Tests for Arabic character counting."""

    def test_pure_arabic(self):
        # "مادة" = 4 Arabic chars
        assert count_arabic_chars("مادة") == 4

    def test_mixed(self):
        # "المادة 3" = 6 Arabic chars (ا ل م ا د ة)
        assert count_arabic_chars("المادة 3") == 6

    def test_empty(self):
        assert count_arabic_chars("") == 0


class TestExtractLawReference:
    """Tests for structured law reference extraction."""

    def test_federal_law_extraction(self):
        text = "قانون اتحادي رقم (3) لسنة 2022 بشأن تنظيم الاتصالات"
        ref = extract_law_reference(text)
        assert ref is not None
        assert ref["doc_type"] == "قانون"
        assert ref["number"] == "3"
        assert ref["year"] == "2022"

    def test_decree_extraction(self):
        text = "مرسوم رقم (15) لسنة 2021 بإنشاء الهيئة"
        ref = extract_law_reference(text)
        assert ref is not None
        assert ref["doc_type"] == "مرسوم"
        assert ref["number"] == "15"

    def test_returns_none_when_no_match(self):
        text = "هذا النص لا يحتوي على رقم قانون"
        assert extract_law_reference(text) is None

    def test_extracts_from_multiline_text(self):
        """Should find the reference even if it appears in the middle of a document."""
        text = (
            "ديباجة القانون\n"
            "نحن رئيس الدولة،\n"
            "بناءً على قانون اتحادي رقم (5) لسنة 2023\n"
            "الباب الأول"
        )
        ref = extract_law_reference(text)
        assert ref is not None
        assert ref["year"] == "2023"
