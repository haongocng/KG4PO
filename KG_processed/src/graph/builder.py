from __future__ import annotations

import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from src.data.loader import LoadedData, ProductInfo, SessionSample
from .triplet_extractor import TripletExtractor, Triplet, NodeType
from .co_occur_handler import CoOccurHandler, CoOccurConfig


# ---------------------------------------------------------------------------
# CKGraph  —  the Collaborative Knowledge Graph data structure
# ---------------------------------------------------------------------------

@dataclass
class CKGraph:
    """
    In-memory representation of the Collaborative Knowledge Graph.

    Attributes
    ----------
    triplets       : full list of (head, relation, tail) edges
    node_types     : node_id  -> NodeType string
    adjacency      : head_id  -> list of (relation, tail_id)  outgoing edges
    reverse_adj    : tail_id  -> list of (relation, head_id)  incoming edges
    all_nodes      : set of every node id in the graph
    all_relations  : set of every relation type in the graph
    item_nodes     : set of node ids whose type == NodeType.ITEM
    session_nodes  : set of node ids whose type == NodeType.SESSION
    category_nodes : set of node ids whose type == NodeType.CATEGORY
    """
    triplets:        list[Triplet]               = field(default_factory=list)
    node_types:      dict[str, str]              = field(default_factory=dict)
    adjacency:       dict[str, list[tuple[str, str]]] = field(
                         default_factory=lambda: defaultdict(list)
                     )
    reverse_adj:     dict[str, list[tuple[str, str]]] = field(
                         default_factory=lambda: defaultdict(list)
                     )
    all_nodes:       set[str]                    = field(default_factory=set)
    all_relations:   set[str]                    = field(default_factory=set)
    item_nodes:      set[str]                    = field(default_factory=set)
    session_nodes:   set[str]                    = field(default_factory=set)
    category_nodes:  set[str]                    = field(default_factory=set)

    # -------------------------------------------------------------------
    # Convenience query methods
    # -------------------------------------------------------------------

    def neighbors(self, node_id: str) -> list[tuple[str, str]]:
        """Return list of (relation, tail_id) for outgoing edges from node."""
        return self.adjacency.get(node_id, [])

    def in_neighbors(self, node_id: str) -> list[tuple[str, str]]:
        """Return list of (relation, head_id) for incoming edges to node."""
        return self.reverse_adj.get(node_id, [])

    def node_type(self, node_id: str) -> str:
        return self.node_types.get(node_id, NodeType.ITEM)

    def stats(self) -> dict[str, int]:
        """Return basic graph statistics."""
        return {
            "num_nodes":     len(self.all_nodes),
            "num_triplets":  len(self.triplets),
            "num_relations": len(self.all_relations),
            "num_items":     len(self.item_nodes),
            "num_sessions":  len(self.session_nodes),
            "num_categories": len(self.category_nodes),
        }

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the graph to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "CKGraph":
        """Deserialise a previously saved graph."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected CKGraph, got {type(obj)}")
        return obj


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------

class GraphBuilder:
    """
    Orchestrates TripletExtractor and CoOccurHandler to build a CKGraph
    from a LoadedData object.

    Usage
    -----
        builder = GraphBuilder()
        graph   = builder.build(loaded_data)
    """

    def __init__(
        self,
        co_occur_config: CoOccurConfig | None = None,
        include_description: bool = True,
        include_keywords: bool = True,
    ):
        """
        Args:
            co_occur_config     : settings for co-occurrence edge generation
            include_description : forward to TripletExtractor
            include_keywords    : forward to TripletExtractor
        """
        self._co_cfg             = co_occur_config or CoOccurConfig()
        self._include_desc       = include_description
        self._include_kw         = include_keywords

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def build(self, data: LoadedData) -> CKGraph:
        """
        Build the full CKG from a LoadedData object.

        Steps:
          1. Extract knowledge triplets from product info
          2. Extract co-occurrence triplets from sessions
          3. Merge all triplets + node types
          4. Build adjacency indexes
          5. Populate convenience node-type sets

        Args:
            data: output of DataLoader.load()

        Returns:
            CKGraph ready for embedding and propagation
        """
        # Step 1 – knowledge triplets
        extractor  = TripletExtractor(
            include_description=self._include_desc,
            include_keywords=self._include_kw,
        )
        kg_triplets  = extractor.extract_all(data.products)
        kg_types     = extractor.get_node_types()

        # Step 2 – co-occurrence triplets
        co_handler   = CoOccurHandler(config=self._co_cfg)
        co_triplets  = co_handler.extract_all(data.samples)
        co_types     = co_handler.get_node_types()

        # Step 3 – merge
        all_triplets = kg_triplets + co_triplets
        all_types    = {**kg_types, **co_types}

        # Step 4 & 5 – assemble graph
        graph = self._assemble(all_triplets, all_types)

        print(
            f"[GraphBuilder] CKG built — "
            + ", ".join(f"{k}={v}" for k, v in graph.stats().items())
        )
        return graph

    def build_from_parts(
        self,
        products: dict[str, ProductInfo],
        samples: list[SessionSample],
    ) -> CKGraph:
        """
        Alternative entry point when products and samples are available
        separately (without a full LoadedData wrapper).
        """
        from src.data.loader import LoadedData
        from src.data.schema_detector import SchemaDetector

        dummy = LoadedData(
            samples=samples,
            products=products,
            corpus_schema=SchemaDetector.detect_corpus(
                [p.raw for p in products.values()]
            ),
        )
        return self.build(dummy)

    # -------------------------------------------------------------------
    # Internal assembly
    # -------------------------------------------------------------------

    def _assemble(
        self,
        triplets: list[Triplet],
        node_types: dict[str, str],
    ) -> CKGraph:
        adjacency:   dict[str, list[tuple[str, str]]] = defaultdict(list)
        reverse_adj: dict[str, list[tuple[str, str]]] = defaultdict(list)
        all_nodes:   set[str] = set()
        all_rels:    set[str] = set()

        for t in triplets:
            adjacency[t.head].append((t.relation, t.tail))
            reverse_adj[t.tail].append((t.relation, t.head))
            all_nodes.add(t.head)
            all_nodes.add(t.tail)
            all_rels.add(t.relation)

        # Ensure every node has an entry (even leaf nodes with no outgoing edge)
        for node in all_nodes:
            if node not in adjacency:
                adjacency[node] = []
            if node not in reverse_adj:
                reverse_adj[node] = []

        # Register any nodes not yet in node_types (e.g. tail-only nodes)
        for node in all_nodes:
            if node not in node_types:
                node_types[node] = self._infer_type_from_id(node)

        # Convenience sets grouped by type
        item_nodes     = {n for n, t in node_types.items() if t == NodeType.ITEM}
        session_nodes  = {n for n, t in node_types.items() if t == NodeType.SESSION}
        category_nodes = {n for n, t in node_types.items() if t == NodeType.CATEGORY}

        return CKGraph(
            triplets=triplets,
            node_types=node_types,
            adjacency=dict(adjacency),
            reverse_adj=dict(reverse_adj),
            all_nodes=all_nodes,
            all_relations=all_rels,
            item_nodes=item_nodes,
            session_nodes=session_nodes,
            category_nodes=category_nodes,
        )

    @staticmethod
    def _infer_type_from_id(node_id: str) -> str:
        """
        Fallback type inference from node id prefix convention:
          'item::'    -> ITEM
          'session::' -> SESSION
          'cat::'     -> CATEGORY
          'desc::'    -> DESCRIPTION
          'kw::'      -> KEYWORD
        """
        if node_id.startswith("item::"):
            return NodeType.ITEM
        if node_id.startswith("session::"):
            return NodeType.SESSION
        if node_id.startswith("cat::"):
            return NodeType.CATEGORY
        if node_id.startswith("desc::"):
            return NodeType.DESCRIPTION
        if node_id.startswith("kw::"):
            return NodeType.KEYWORD
        return NodeType.ITEM