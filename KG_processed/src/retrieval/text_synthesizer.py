from __future__ import annotations

from dataclasses import dataclass

from src.graph.builder import CKGraph
from src.data.loader import ProductInfo
from .topk_selector import TopKResult, TopKSelector


# ---------------------------------------------------------------------------
# KnowledgeEntry
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeEntry:
    """
    knowledge_retrieve output for a single item in a session.

    Attributes
    ----------
    title              : human-readable item title
    item_id            : graph node id
    knowledge_retrieve : synthesised context snippet for LLM consumption
    """
    title:              str
    item_id:            str
    knowledge_retrieve: str

    def to_dict(self) -> dict:
        return {
            "title":              self.title,
            "item_id":            self.item_id,
            "knowledge_retrieve": self.knowledge_retrieve,
        }


# ---------------------------------------------------------------------------
# TextSynthesizer
# ---------------------------------------------------------------------------

class TextSynthesizer:
    """
    Convert TopKResult objects into natural-language knowledge_retrieve snippets
    and assemble them into a single per-session retrieved_knowledge string.

    Per-item snippet template (generic across domains)
    --------------------------------------------------
    "- Item '<title>' [belong to categories: <C1>, <C2>]
     [with keywords: <K1>, <K2>]
     [related items (inferred): <I1>, <I2>]."

    Each clause is omitted when the relevant data is absent.
    The output is driven entirely by graph attention results (TopKResult),
    NOT by directly reading from ProductInfo — ensuring genuine inference.

    ProductInfo is used only as a last-resort fallback when the graph
    has no attention data for a node (e.g. items missing from info file).
    """

    def __init__(
        self,
        graph:    CKGraph,
        products: dict[str, ProductInfo],
    ):
        self.graph    = graph
        self.products = products

    # -------------------------------------------------------------------
    # Per-item synthesis  (used internally and for per-item output)
    # -------------------------------------------------------------------

    def synthesize(self, topk: TopKResult) -> KnowledgeEntry:
        """
        Build a knowledge_retrieve snippet for a single item from its TopKResult.
        All content is derived from graph attention — no direct info-file reads.
        """
        title = TopKSelector.readable_name(topk.item_id)
        snippet = self._build_item_snippet(title, topk)
        return KnowledgeEntry(
            title=title,
            item_id=topk.item_id,
            knowledge_retrieve=snippet,
        )

    def synthesize_batch(
        self,
        topk_results: dict[str, TopKResult],
    ) -> list[KnowledgeEntry]:
        return [self.synthesize(topk) for topk in topk_results.values()]

    # -------------------------------------------------------------------
    # Per-session assembly  (main output used by pipeline)
    # -------------------------------------------------------------------

    def build_session_knowledge(
        self,
        session_titles:   list[str],
        topk_results:     dict[str, TopKResult],
    ) -> str:
        """
        Assemble one retrieved_knowledge string for the entire session.

        Each item in the session contributes one bullet line.
        Items with no graph coverage get a minimal fallback line.

        Args:
            session_titles  : ordered list of item titles in the session
            topk_results    : dict mapping item_id -> TopKResult

        Returns:
            single string with all item snippets joined by newlines
        """
        lines: list[str] = []
        for title in session_titles:
            item_id = f"item::{title}"
            topk    = topk_results.get(item_id)
            if topk is not None:
                line = self._build_item_snippet(title, topk)
            else:
                line = self._fallback_snippet(title)
            lines.append(line)
        return " ".join(lines)

    # -------------------------------------------------------------------
    # Item snippet builder  (graph-attention driven)
    # -------------------------------------------------------------------

    def _build_item_snippet(self, title: str, topk: TopKResult) -> str:
        """
        Build one bullet snippet for an item entirely from TopKResult.

        Format:
            "- Item '<title>'[ belong to categories: C1, C2][ with keywords: K1, K2]
             [ related items (inferred): I1, I2]."
        """
        parts: list[str] = [f"- Item '{title}'"]

        # --- Categories (from graph attention, most salient first) ---
        if topk.top_categories:
            # Use only the most specific category (highest attention weight)
            # which is typically the deepest taxonomy level
            cat_names = [
                TopKSelector.readable_name(nid)
                for nid, _ in topk.top_categories
            ]
            # Pick the most specific (last level) by finding shortest path
            # heuristic: longer/more specific names are deeper in taxonomy
            most_specific = max(cat_names, key=len)
            # Include up to 2 categories: the most specific + highest weight
            ordered = [most_specific] + [c for c in cat_names if c != most_specific]
            display_cats = ordered[:4]
            parts.append(f"belong to categories: {', '.join(display_cats)}")

        # --- Keywords (from graph attention on keyword nodes) ---
        if topk.top_keywords:
            kw_names = [
                TopKSelector.readable_name(nid)
                for nid, _ in topk.top_keywords
            ]
            parts.append(f"with keywords: {', '.join(kw_names)}")
        else:
            # Fallback: try to get keywords from ProductInfo only if
            # graph returned nothing (e.g. item has no keyword nodes)
            product = self._lookup_product(title)
            if product and product.keywords:
                kw_names = product.keywords[:3]
                parts.append(f"with keywords: {', '.join(kw_names)}")

        # --- Inferred items (genuine graph inference via 2-hop paths) ---
        if topk.top_inferred_items:
            inferred_names = [
                TopKSelector.readable_name(nid)
                for nid, _ in topk.top_inferred_items
            ]
            parts.append(f"related items: {', '.join(inferred_names)}")

        # Join with ". " and close
        if len(parts) == 1:
            return parts[0] + "."

        head   = parts[0]
        clauses = "; ".join(parts[1:])
        return f"{head} {clauses}."

    def _fallback_snippet(self, title: str) -> str:
        """
        Minimal snippet for items not present in the graph.
        Tries ProductInfo first, otherwise returns a bare label.
        """
        product = self._lookup_product(title)
        if product:
            parts: list[str] = [f"- Item '{title}'"]
            if product.categories:
                parts.append(f"belong to categories: {', '.join(product.categories[:2])}")
            elif product.taxonomy_levels:
                vals = list(product.taxonomy_levels.values())
                parts.append(f"belong to categories: {vals[-1]}")
            if product.keywords:
                parts.append(f"with keywords: {', '.join(product.keywords[:3])}")
            if len(parts) == 1:
                return parts[0] + "."
            return f"{parts[0]} {'; '.join(parts[1:])}."
        return f"- Item '{title}'."

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _lookup_product(self, title: str) -> ProductInfo | None:
        return self.products.get(title.strip().lower())