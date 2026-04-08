from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

from src.attention.propagation import PropagationResult
from src.graph.builder import CKGraph
from src.graph.triplet_extractor import NodeType


# ---------------------------------------------------------------------------
# TopKResult
# ---------------------------------------------------------------------------

@dataclass
class TopKResult:
    """
    Top-K attended neighbours for a single item node after propagation.

    Attributes
    ----------
    item_id              : graph node id of the focal item
    top_categories       : top category nodes by aggregated attention weight
    top_keywords         : top keyword nodes by aggregated attention weight
    top_inferred_items   : items NOT in the current session but reachable via
                           shared category/keyword nodes (true graph inference)
    top_descriptions     : description nodes
    all_top              : union of categories + keywords sorted by weight
    """
    item_id:            str
    top_categories:     list[tuple[str, float]] = field(default_factory=list)
    top_keywords:       list[tuple[str, float]] = field(default_factory=list)
    top_inferred_items: list[tuple[str, float]] = field(default_factory=list)
    top_descriptions:   list[tuple[str, float]] = field(default_factory=list)
    all_top:            list[tuple[str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TopKSelector
# ---------------------------------------------------------------------------

class TopKSelector:
    """
    Aggregate attention weights from propagation layers and select top-K
    neighbours per type, including inferred items discovered via shared
    category/keyword bridge nodes (genuine graph inference).

    Inference mechanism
    -------------------
    For a focal item H, inferred items are items I such that:
        H --[belongs_to/has_keyword]--> bridge_node <--[belongs_to/has_keyword]-- I

    The inferred weight of I from H is:
        w(H -> bridge) * w(bridge -> I)   (product of attention weights)

    This captures the KGAT high-order connectivity signal: two items that
    share a category or keyword become connected through the bridge node,
    and the propagation attention tells us how salient that connection is.
    """

    def __init__(
        self,
        graph:              CKGraph,
        k_categories:       int   = 4,
        k_keywords:         int   = 3,
        k_inferred_items:   int   = 3,
        k_descriptions:     int   = 1,
        layer_decay:        float = 0.9,
        exclude_self:       bool  = True,
        session_item_ids:   set[str] | None = None,
    ):
        self.graph            = graph
        self.k_categories     = k_categories
        self.k_keywords       = k_keywords
        self.k_inferred_items = k_inferred_items
        self.k_descriptions   = k_descriptions
        self.layer_decay      = layer_decay
        self.exclude_self     = exclude_self
        self.session_item_ids = session_item_ids or set()
        self._bridge_to_items: dict[str, list[str]] = self._build_bridge_index()

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def select(
        self,
        item_id: str,
        result:  PropagationResult,
    ) -> TopKResult:
        direct_weights   = self._aggregate_direct(item_id, result)
        inferred_weights = self._infer_items(item_id, direct_weights, result)
        return self._build_result(item_id, direct_weights, inferred_weights)

    def select_batch(
        self,
        item_ids: list[str],
        result:   PropagationResult,
    ) -> dict[str, TopKResult]:
        return {iid: self.select(iid, result) for iid in item_ids}

    # -------------------------------------------------------------------
    # Direct weight aggregation
    # -------------------------------------------------------------------

    def _aggregate_direct(
        self,
        head_id: str,
        result:  PropagationResult,
    ) -> dict[str, float]:
        """
        Sum attention weights for each (head_id -> tail) pair across all
        propagation layers with optional layer decay.
        """
        accumulated: dict[str, float] = defaultdict(float)
        for layer_idx, head_map in result.attention_records.items():
            decay = self.layer_decay ** layer_idx
            for _rel, tail_id, weight in head_map.get(head_id, []):
                if self.exclude_self and tail_id == head_id:
                    continue
                accumulated[tail_id] += weight * decay
        return dict(accumulated)

    # -------------------------------------------------------------------
    # Inferred item discovery  (2-hop: item -> bridge -> other_item)
    # -------------------------------------------------------------------

    def _infer_items(
        self,
        focal_id:       str,
        direct_weights: dict[str, float],
        result:         PropagationResult,
    ) -> dict[str, float]:
        """
        Discover items connected to focal_id through shared bridge nodes
        (categories and keywords) using attention-weighted 2-hop paths.

        For each bridge node B attended by focal_id with weight w1:
          For each item I connected to B via reverse adjacency:
            inferred_score[I] += w1 * attention(B -> I at layer 1)

        Items already in the current session are excluded.
        """
        inferred: dict[str, float] = defaultdict(float)

        bridge_weights: dict[str, float] = {
            nid: w
            for nid, w in direct_weights.items()
            if self.graph.node_type(nid) in (NodeType.CATEGORY, NodeType.KEYWORD)
        }

        # Attention weights FROM bridge nodes at layer 1
        layer1_records = result.attention_records.get(1, {})

        for bridge_id, w_focal_to_bridge in bridge_weights.items():
            candidate_items = self._bridge_to_items.get(bridge_id, [])

            bridge_neighbors = {
                tail: w
                for _rel, tail, w in layer1_records.get(bridge_id, [])
                if self.graph.node_type(tail) == NodeType.ITEM
            }

            for item_id in candidate_items:
                if item_id == focal_id:
                    continue
                if item_id in self.session_item_ids:
                    continue
                w_bridge_to_item = bridge_neighbors.get(item_id, 0.0)
                if w_bridge_to_item > 0:
                    inferred[item_id] += w_focal_to_bridge * w_bridge_to_item
                else:
                    # Weak fallback signal when bridge was not attended at layer 1
                    inferred[item_id] += w_focal_to_bridge * 0.01

        return dict(inferred)

    # -------------------------------------------------------------------
    # Build TopKResult
    # -------------------------------------------------------------------

    def _build_result(
        self,
        item_id:          str,
        direct_weights:   dict[str, float],
        inferred_weights: dict[str, float],
    ) -> TopKResult:
        categories:   list[tuple[str, float]] = []
        keywords:     list[tuple[str, float]] = []
        descriptions: list[tuple[str, float]] = []

        for node_id, weight in direct_weights.items():
            ntype = self.graph.node_type(node_id)
            if ntype == NodeType.CATEGORY:
                categories.append((node_id, weight))
            elif ntype == NodeType.KEYWORD:
                keywords.append((node_id, weight))
            elif ntype == NodeType.DESCRIPTION:
                descriptions.append((node_id, weight))

        categories   = self._topk(categories,   self.k_categories)
        keywords     = self._topk(keywords,      self.k_keywords)
        descriptions = self._topk(descriptions, self.k_descriptions)

        inferred_items = self._topk(
            list(inferred_weights.items()),
            self.k_inferred_items,
        )

        all_top = self._topk(
            categories + keywords,
            k=self.k_categories + self.k_keywords,
        )

        return TopKResult(
            item_id=item_id,
            top_categories=categories,
            top_keywords=keywords,
            top_inferred_items=inferred_items,
            top_descriptions=descriptions,
            all_top=all_top,
        )

    # -------------------------------------------------------------------
    # Bridge index
    # -------------------------------------------------------------------

    def _build_bridge_index(self) -> dict[str, list[str]]:
        """
        Build reverse index: bridge_node_id -> [item_ids connected to it].
        Bridge nodes are CATEGORY and KEYWORD nodes.
        """
        index: dict[str, list[str]] = defaultdict(list)
        for node_id in self.graph.item_nodes:
            for _rel, tail_id in self.graph.neighbors(node_id):
                ntype = self.graph.node_type(tail_id)
                if ntype in (NodeType.CATEGORY, NodeType.KEYWORD):
                    index[tail_id].append(node_id)
        return dict(index)

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _topk(
        pairs: list[tuple[str, float]],
        k:     int,
    ) -> list[tuple[str, float]]:
        return sorted(pairs, key=lambda x: x[1], reverse=True)[:k]

    @staticmethod
    def readable_name(node_id: str) -> str:
        """Strip type prefix to get human-readable label."""
        name = node_id
        for prefix in ("item::", "cat::", "kw::", "desc::item::", "desc::", "session::"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        return name