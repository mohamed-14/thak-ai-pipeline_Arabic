"""
src/pipeline/structural_extractor.py
──────────────────────────────────────
Stage 3 of the v2 Pipeline: Structural Extraction.

This stage uses the structural vocabulary profile produced by Stage 2 to
walk through the full normalised document text and build a tree of nodes
representing the legal hierarchy (Chapters → Articles → Clauses).

Design philosophy: hybrid LLM + regex.
─────────────────────────────────────
The profiler (Stage 2) is the intelligent part — it uses the LLM to
discover *what patterns exist* in this document.  Stage 3 is the efficient
part — it uses the regex patterns that the LLM produced to do the actual
line-by-line parsing.  This means we only pay LLM cost once per document
(during profiling), and then do the bulk extraction cheaply with regex.

This approach is far more reliable and cost-effective than asking an LLM
to extract every single article from a 50-page document in one shot.

The extractor builds a flat list of NodeRecord objects that the warehouse
stores in `document_nodes`.  The parent-child tree is represented via the
`parent_id` field (standard adjacency-list pattern) so it can be
reconstructed with a recursive CTE query when needed.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID, uuid4

from src.database.warehouse import WarehouseClient
from src.utils.arabic_utils import (
    ARTICLE_PATTERN,
    CHAPTER_PATTERN,
    CLAUSE_PATTERN,
    CLOSING_PATTERN,
    PREAMBLE_PATTERN,
)

logger = logging.getLogger(__name__)


@dataclass
class NodeRecord:
    """
    A single node in the extracted legal document hierarchy.
    Flat data class that maps directly to a document_nodes DB row.
    """
    id: UUID = field(default_factory=uuid4)
    parent_id: Optional[UUID] = None
    node_type: str = "article"           # chapter | article | clause | preamble
    node_type_arabic: Optional[str] = None
    node_number: Optional[str] = None
    depth: int = 0
    heading: Optional[str] = None
    text_content: str = ""
    sequence_index: int = 0


class StructuralExtractor:
    """
    Parses the full normalised document text into a hierarchy of NodeRecords
    using the regex patterns discovered by the DocumentProfiler.
    """

    def __init__(self, warehouse: WarehouseClient) -> None:
        self._db = warehouse

    def extract_and_store(
        self,
        document_id: UUID,
        text_clean: str,
        profile: dict,
    ) -> list[NodeRecord]:
        """
        Extract the document hierarchy and write it to `document_nodes`.

        Parameters
        ----------
        document_id : UUID
        text_clean : str
            Normalised full document text (text_clean from documents_raw).
        profile : dict
            The structural profile returned by DocumentProfiler.profile().

        Returns
        -------
        list[NodeRecord]
            Flat list of all extracted nodes, ordered by their appearance
            in the document.
        """
        # Build compiled regex patterns from the profile vocabulary.
        # We compile them here rather than in the profiler because Python
        # regex objects are not JSON-serialisable.
        compiled_patterns = self._compile_patterns(profile)

        nodes = self._parse_hierarchy(text_clean, compiled_patterns, profile)

        # Convert to dicts for the warehouse
        node_dicts = [
            {
                "id": n.id,
                "parent_id": n.parent_id,
                "node_type": n.node_type,
                "node_type_arabic": n.node_type_arabic,
                "node_number": n.node_number,
                "depth": n.depth,
                "heading": n.heading,
                "text_content": n.text_content,
                "sequence_index": n.sequence_index,
            }
            for n in nodes
        ]
        self._db.store_nodes(document_id, node_dicts)
        self._db.mark_document_status(document_id, "structured")

        logger.info(
            "Document %s: extracted %d structural nodes (chapters=%d, articles=%d)",
            document_id,
            len(nodes),
            sum(1 for n in nodes if n.depth == 0),
            sum(1 for n in nodes if n.depth == 1),
        )
        return nodes

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _compile_patterns(profile: dict) -> list[dict]:
        """
        Compile the regex strings from the profile vocabulary into Python
        regex objects, sorted by depth (so we try higher-level patterns first).

        Falls back to built-in patterns from arabic_utils if the profile
        vocabulary is empty or missing.
        """
        vocab = profile.get("structural_vocabulary") or []
        compiled = []

        for item in vocab:
            raw_regex = item.get("regex", "")
            if not raw_regex:
                continue
            try:
                compiled.append(
                    {
                        "pattern": re.compile(raw_regex, re.MULTILINE | re.UNICODE),
                        "node_type": item.get("division_type", "unknown").lower(),
                        "node_type_arabic": item.get("arabic_term", ""),
                        "depth": item.get("level", 1) - 1,  # convert 1-based to 0-based
                    }
                )
            except re.error as exc:
                logger.warning(
                    "Invalid regex from profiler for '%s': %s — using fallback.",
                    item.get("division_type"),
                    exc,
                )

        # Always ensure we have at least chapter and article patterns
        if not any(p["depth"] == 0 for p in compiled):
            compiled.append(
                {
                    "pattern": CHAPTER_PATTERN,
                    "node_type": "chapter",
                    "node_type_arabic": "الباب",
                    "depth": 0,
                }
            )
        if not any(p["depth"] == 1 for p in compiled):
            compiled.append(
                {
                    "pattern": ARTICLE_PATTERN,
                    "node_type": "article",
                    "node_type_arabic": "المادة",
                    "depth": 1,
                }
            )

        return sorted(compiled, key=lambda x: x["depth"])

    def _parse_hierarchy(
        self,
        text: str,
        patterns: list[dict],
        profile: dict,
    ) -> list[NodeRecord]:
        """
        Main parsing loop.  Walks line-by-line through the document and
        assigns each line to a structural node based on regex matches.

        The parser maintains a stack of "open" parent nodes so that when
        a match occurs at a given depth, all deeper open nodes are closed
        and the new node is attached to the correct parent.
        """
        lines = text.splitlines()
        nodes: list[NodeRecord] = []

        # Stack tracks the current open parent at each depth level.
        # parent_stack[0] = current chapter, [1] = current article, etc.
        parent_stack: list[Optional[NodeRecord]] = [None] * 4
        current_node: Optional[NodeRecord] = None
        current_depth = -1
        seq_counters: dict[int, int] = {}  # depth → running sequence count

        # Handle preamble: collect text before the first structural marker
        preamble_lines: list[str] = []
        preamble_done = False

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_node is not None:
                    current_node.text_content += "\n"
                continue

            # Check for document closing (e.g. signature block)
            if CLOSING_PATTERN.search(stripped):
                if current_node is not None:
                    current_node.text_content += "\n" + stripped
                # We could mark remaining lines as a "closing" node, but for
                # legal analysis purposes the closing block has no substantive
                # content so we just append it to the last open node.
                continue

            # Try to match each structural pattern against this line
            matched = False
            for pattern_info in patterns:
                if pattern_info["pattern"].match(stripped):
                    matched = True
                    depth = pattern_info["depth"]

                    # Close current node (save it to nodes list)
                    if current_node is not None and current_node.text_content.strip():
                        nodes.append(current_node)

                    # Determine parent from the stack
                    parent: Optional[NodeRecord] = None
                    if depth > 0:
                        for d in range(depth - 1, -1, -1):
                            if parent_stack[d] is not None:
                                parent = parent_stack[d]
                                break

                    # Create the new node
                    seq_counters[depth] = seq_counters.get(depth, -1) + 1
                    # Reset deeper counters when a new node at this level starts
                    for d in range(depth + 1, 4):
                        seq_counters[d] = -1
                        parent_stack[d] = None

                    new_node = NodeRecord(
                        parent_id=parent.id if parent else None,
                        node_type=pattern_info["node_type"],
                        node_type_arabic=pattern_info["node_type_arabic"],
                        depth=depth,
                        heading=stripped,
                        text_content=stripped,
                        sequence_index=seq_counters[depth],
                    )
                    # Extract the node number from the heading
                    new_node.node_number = self._extract_number(stripped)

                    parent_stack[depth] = new_node
                    current_node = new_node
                    current_depth = depth
                    preamble_done = True
                    break

            if not matched:
                # Line belongs to the current open node (or preamble)
                if preamble_done and current_node is not None:
                    current_node.text_content += "\n" + stripped
                elif not preamble_done:
                    preamble_lines.append(stripped)

        # Flush the last open node
        if current_node is not None and current_node.text_content.strip():
            nodes.append(current_node)

        # Create a preamble node if we collected preamble lines
        if preamble_lines:
            preamble_text = "\n".join(preamble_lines).strip()
            if preamble_text:
                preamble_node = NodeRecord(
                    node_type="preamble",
                    node_type_arabic="ديباجة",
                    depth=0,
                    heading="ديباجة",
                    text_content=preamble_text,
                    sequence_index=0,
                )
                nodes.insert(0, preamble_node)

        return nodes

    @staticmethod
    def _extract_number(heading: str) -> Optional[str]:
        """
        Extract the number or ordinal from a heading line.
        E.g. "المادة (5)" → "5", "الباب الثالث" → "الثالث".
        """
        # Try Western numerals first (most common after our digit normalisation)
        m = re.search(r"\(?(\d+)\)?", heading)
        if m:
            return m.group(1)
        # Try Arabic ordinal words
        ordinals = [
            "الأول", "الثاني", "الثالث", "الرابع", "الخامس",
            "السادس", "السابع", "الثامن", "التاسع", "العاشر",
        ]
        for ordinal in ordinals:
            if ordinal in heading:
                return ordinal
        return None
