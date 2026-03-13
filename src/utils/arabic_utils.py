"""
src/utils/arabic_utils.py
──────────────────────────
Arabic-specific text processing utilities.

Arabic text extracted from PDFs comes with many normalisation challenges:
  - Multiple Unicode representations of the same letter (e.g. alef variants)
  - Harakat (short vowel diacritics / tashkeel) added in formal legal text
  - Tatweel (kashida) stretching used for typographic alignment
  - Mixed Arabic-Indic (٠١٢...) and Western (0123...) numeral systems
  - Ligatures and presentation forms that differ from canonical Arabic

These helpers ensure consistent, clean text before any LLM or regex step.
"""

import re
import unicodedata
from typing import Optional


# ── Unicode codepoint constants ───────────────────────────────────────────────

# Harakat: short vowels and other diacritics
_HARAKAT = re.compile(
    r"[\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)

# Tatweel / kashida elongation character
_TATWEEL = re.compile(r"\u0640")

# Alef variants → normalise to plain alef (ا)
_ALEF_VARIANTS = re.compile(r"[\u0622\u0623\u0625\u0671]")  # أ إ آ ٱ

# Teh Marbuta → Ha (context-independent normalisation)
_TEH_MARBUTA = re.compile(r"\u0629")  # ة → ه

# Waw with Hamza variants
_WAW_HAMZA = re.compile(r"\u0624")  # ؤ → و

# Ya variants (dotless ya / alef maqsura)
_ALEF_MAQSURA = re.compile(r"\u0649")  # ى → ي

# Arabic-Indic numerals to Western Arabic numerals
_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Extended Arabic-Indic (Farsi/Urdu) variants
_EXTENDED_ARABIC_INDIC = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# Presentation forms: Arabic Presentation Forms-A (FB50–FDFF)
# and Arabic Presentation Forms-B (FE70–FEFF) should be NFKC-normalised.
# We handle this via unicodedata.normalize("NFKC", ...) below.


# ── Structural vocabulary patterns for UAE legislation ────────────────────────
# These patterns cover the most common hierarchical markers used in federal laws,
# government decrees, and regulatory publications from the UAE.

# Example matches: "الباب الأول", "الباب الثاني", "الباب (1)"
CHAPTER_PATTERN = re.compile(
    r"^[\s\u200f]*"                      # optional leading whitespace / RLM
    r"(الباب|الفصل|القسم)"               # chapter keywords
    r"\s+"
    r"("
    r"[الأولىالثانيالثالثالرابعالخامسالسادسالسابعالثامنالتاسعالعاشر]+"  # ordinals
    r"|[\u0600-\u06FF\u0660-\u0669]+"   # any Arabic word/numeral
    r"|\(\d+\)|\d+"                      # Western numeral, possibly parenthesised
    r")",
    re.MULTILINE | re.UNICODE,
)

# Example matches: "المادة (1)", "المادة الأولى", "مادة 5"
ARTICLE_PATTERN = re.compile(
    r"^[\s\u200f]*"
    r"(المادة|مادة)"
    r"[\s\u00a0]+"
    r"("
    r"\(\d+\)"                           # (1)
    r"|\d+"                              # 1
    r"|[الأولىالثانيالثالثالرابعالخامس]+"  # الأولى etc.
    r")",
    re.MULTILINE | re.UNICODE,
)

# Example matches: "أولاً:", "ثانياً:", "ثالثاً:"
ORDINAL_CLAUSE_PATTERN = re.compile(
    r"^[\s\u200f]*"
    r"(أولاً|ثانياً|ثالثاً|رابعاً|خامساً|سادساً|سابعاً|ثامناً|تاسعاً|عاشراً)"
    r"[\s:–\-]",
    re.MULTILINE | re.UNICODE,
)

# Example matches: "1-", "أ-", "(أ)", "(1)"
CLAUSE_PATTERN = re.compile(
    r"^[\s\u200f]*"
    r"(\(\s*[\dأ-ي]\s*\)|\d+[\-\.\)]\s|[أ-ي][\-\.\)]\s)",
    re.MULTILINE | re.UNICODE,
)

# Preamble / recital block
PREAMBLE_PATTERN = re.compile(
    r"(بناءً على|استناداً إلى|إستناداً|وفقاً لأحكام|نحن.+حاكم)",
    re.MULTILINE | re.UNICODE,
)

# Signature / closing block (end of substantive content)
CLOSING_PATTERN = re.compile(
    r"(صدر في|يُنشر هذا|يُعمل بهذا|والله الموفق)",
    re.MULTILINE | re.UNICODE,
)

# Law number in title lines, e.g. "قانون اتحادي رقم (3) لسنة 2022"
LAW_NUMBER_PATTERN = re.compile(
    r"(قانون|مرسوم|قرار)\s+.{0,30}?\s*رقم\s*[(\[]?\s*(\d+)\s*[)\]]?"
    r"\s+لسنة\s+(\d{4})",
    re.UNICODE,
)


# ── Core normalisation function ───────────────────────────────────────────────

def normalize_arabic(text: str, aggressive: bool = False) -> str:
    """
    Normalise Arabic text for consistent downstream processing.

    Parameters
    ----------
    text : str
        Raw Arabic text as extracted from PDF or OCR.
    aggressive : bool
        If True, also normalises teh marbuta and alef maqsura — useful for
        search/matching, but may reduce faithfulness for display purposes.
        Default is False (safer for legal text where exact spelling matters).

    Returns
    -------
    str
        Cleaned, normalised Arabic string.
    """
    if not text:
        return text

    # Step 1: NFKC normalisation collapses Arabic presentation forms into their
    # canonical Unicode equivalents (e.g. lam-alef ligatures → separate chars).
    text = unicodedata.normalize("NFKC", text)

    # Step 2: Remove harakat (diacritics).  In legal documents they add weight
    # without changing structural meaning and interfere with regex matching.
    text = _HARAKAT.sub("", text)

    # Step 3: Remove tatweel / kashida — purely typographic, no semantic value.
    text = _TATWEEL.sub("", text)

    # Step 4: Normalise alef variants → plain alef (ا).
    # This ensures "أحكام" and "احكام" are treated identically.
    text = _ALEF_VARIANTS.sub("\u0627", text)

    # Step 5: Translate Arabic-Indic digits to Western digits so that numeric
    # patterns in regexes work uniformly.
    text = text.translate(_ARABIC_INDIC_DIGITS)
    text = text.translate(_EXTENDED_ARABIC_INDIC)

    # Step 6 (aggressive only): Further normalise teh marbuta and alef maqsura.
    if aggressive:
        text = _TEH_MARBUTA.sub("\u0647", text)     # ة → ه
        text = _ALEF_MAQSURA.sub("\u064A", text)    # ى → ي

    # Step 7: Collapse multiple whitespace / control chars into single spaces,
    # but preserve line breaks (important for structural line detection).
    text = re.sub(r"[^\S\n]+", " ", text)

    # Step 8: Strip trailing/leading whitespace from each line.
    lines = [line.strip() for line in text.splitlines()]

    # Step 9: Remove completely empty lines that result from PDF extraction
    # artefacts, but keep intentional paragraph breaks (double newlines).
    cleaned_lines: list[str] = []
    consecutive_empty = 0
    for line in lines:
        if line == "":
            consecutive_empty += 1
            if consecutive_empty <= 1:          # allow one blank line through
                cleaned_lines.append(line)
        else:
            consecutive_empty = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def remove_extraction_noise(text: str) -> str:
    """
    Remove common PDF-extraction artefacts that are specific to Arabic legal
    PDFs published on UAE government portals.

    This is intentionally separate from normalize_arabic() so it can be
    applied as a distinct pre-processing step and disabled if needed.
    """
    # Page number artefacts: standalone numerals on their own line
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)

    # Running headers / footers that repeat the law title
    # (handled heuristically: lines under 20 chars that appear > 3 times)
    lines = text.splitlines()
    from collections import Counter
    short_lines = [l.strip() for l in lines if 0 < len(l.strip()) < 20]
    frequent = {l for l, c in Counter(short_lines).items() if c > 3}
    text = "\n".join(l for l in lines if l.strip() not in frequent)

    # Remove null bytes and non-printable characters (except CR/LF/tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    return text


def is_arabic_text(text: str, threshold: float = 0.4) -> bool:
    """
    Return True if the fraction of Arabic-script characters exceeds `threshold`.
    Used to decide whether a page needs OCR or not.
    """
    if not text:
        return False
    total = len(text.replace(" ", "").replace("\n", ""))
    if total == 0:
        return False
    arabic_chars = sum(
        1 for ch in text if "\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F"
    )
    return (arabic_chars / total) >= threshold


def extract_law_reference(text: str) -> Optional[dict]:
    """
    Attempt to extract a structured law reference from the document title
    or preamble, e.g. "قانون اتحادي رقم (3) لسنة 2022".

    Returns a dict with keys: doc_type, number, year — or None if not found.
    """
    match = LAW_NUMBER_PATTERN.search(text)
    if match:
        return {
            "doc_type": match.group(1),
            "number": match.group(2),
            "year": match.group(3),
        }
    return None


def count_arabic_chars(text: str) -> int:
    """Count the number of Arabic-script characters in a string."""
    return sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
