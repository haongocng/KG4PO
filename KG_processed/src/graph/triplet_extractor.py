from dataclasses import dataclass
from typing import Any

from src.data.schema_detector import SchemaDetector, SchemaType
from src.data.loader import ProductInfo


# ---------------------------------------------------------------------------
# Relation constants
# ---------------------------------------------------------------------------

class Relation:
    # Item -> taxonomy entity relations (one per level)
    BELONGS_TO_L1 = "belongs_to_L1"
    BELONGS_TO_L2 = "belongs_to_L2"
    BELONGS_TO_L3 = "belongs_to_L3"
    BELONGS_TO_L4 = "belongs_to_L4"
    BELONGS_TO_LX = "belongs_to_L{}"          # Template for arbitrary levels

    # Item -> flat category entity relation
    BELONGS_TO    = "belongs_to"

    # Item -> description entity relation
    HAS_DESCRIPTION = "has_description"

    # Item -> keyword entity relation
    HAS_KEYWORD   = "has_keyword"

    # Session -> item relation (handled by CoOccurHandler)
    CO_OCCUR      = "co_occur"

    @staticmethod
    def belongs_to_level(level_key: str) -> str:
        """
        Convert a taxonomy level key (e.g. 'Level_3') to a relation name.
        e.g. 'Level_3' -> 'belongs_to_L3'
        """
        try:
            num = level_key.split("_")[1]
            return f"belongs_to_L{num}"
        except (IndexError, ValueError):
            return "belongs_to_Lx"


# ---------------------------------------------------------------------------
# Triplet dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Triplet:
    """
    Immutable triple: (head_node, relation, tail_node).
    All node identifiers are strings.
    """
    head: str
    relation: str
    tail: str

    def __repr__(self) -> str:
        return f"({self.head!r}, {self.relation}, {self.tail!r})"


# ---------------------------------------------------------------------------
# Node type registry
# ---------------------------------------------------------------------------

class NodeType:
    ITEM        = "item"
    SESSION     = "session"
    CATEGORY    = "category"
    DESCRIPTION = "description"
    KEYWORD     = "keyword"


# ---------------------------------------------------------------------------
# TripletExtractor
# ---------------------------------------------------------------------------

class TripletExtractor:
    """
    Extract triplets from ProductInfo objects.

    Produces:
      - (item_title, belongs_to_Lx, taxonomy_entity)  for TAXONOMY schema
      - (item_title, belongs_to,    category_entity)   for CATEGORY schema
      - (item_title, has_description, desc_entity)     if description exists
      - (item_title, has_keyword,   keyword_entity)    if keywords exist

    Also maintains a node_type registry mapping each node_id -> NodeType.
    """

    def __init__(
        self,
        include_description: bool = True,
        include_keywords: bool = True,
        max_description_length: int = 200,
    ):
        """
        Args:
            include_description    : whether to create description nodes
            include_keywords       : whether to create keyword nodes
            max_description_length : truncate description text used as node id
        """
        self.include_description     = include_description
        self.include_keywords        = include_keywords
        self.max_description_length  = max_description_length

        # node_id -> NodeType
        self._node_types: dict[str, str] = {}

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def extract(self, product: ProductInfo) -> list[Triplet]:
        """
        Extract all triplets for a single product.

        Args:
            product: parsed ProductInfo object

        Returns:
            list of Triplet objects
        """
        triplets: list[Triplet] = []
        item_id = self._item_id(product.title)
        self._register_node(item_id, NodeType.ITEM)

        if product.schema_type == SchemaType.TAXONOMY:
            triplets.extend(self._extract_taxonomy(item_id, product.taxonomy_levels))
        elif product.schema_type == SchemaType.CATEGORY:
            triplets.extend(self._extract_categories(item_id, product.categories))

        if self.include_description and product.description:
            triplets.extend(self._extract_description(item_id, product.description))

        if self.include_keywords and product.keywords:
            triplets.extend(self._extract_keywords(item_id, product.keywords))

        return triplets

    def extract_all(self, products: dict[str, ProductInfo]) -> list[Triplet]:
        """
        Extract triplets for all products in the corpus.

        Args:
            products: dict mapping normalized_title -> ProductInfo

        Returns:
            deduplicated list of all Triplet objects
        """
        seen:     set[Triplet]  = set()
        triplets: list[Triplet] = []

        for product in products.values():
            for t in self.extract(product):
                if t not in seen:
                    seen.add(t)
                    triplets.append(t)

        return triplets

    def get_node_types(self) -> dict[str, str]:
        """Return a copy of the node_type registry built so far."""
        return dict(self._node_types)

    # -------------------------------------------------------------------
    # Schema-specific extractors
    # -------------------------------------------------------------------

    def _extract_taxonomy(
        self,
        item_id: str,
        taxonomy_levels: dict[str, str],
    ) -> list[Triplet]:
        """Create one triplet per taxonomy level."""
        triplets = []
        for level_key, level_value in taxonomy_levels.items():
            entity_id = self._category_id(level_value)
            relation  = Relation.belongs_to_level(level_key)
            self._register_node(entity_id, NodeType.CATEGORY)
            triplets.append(Triplet(head=item_id, relation=relation, tail=entity_id))
        return triplets

    def _extract_categories(
        self,
        item_id: str,
        categories: list[str],
    ) -> list[Triplet]:
        """Create one triplet per flat category label."""
        triplets = []
        for cat in categories:
            entity_id = self._category_id(cat)
            self._register_node(entity_id, NodeType.CATEGORY)
            triplets.append(
                Triplet(head=item_id, relation=Relation.BELONGS_TO, tail=entity_id)
            )
        return triplets

    def _extract_description(
        self,
        item_id: str,
        description: str,
    ) -> list[Triplet]:
        """Create a description node linked to the item."""
        truncated  = description[: self.max_description_length].strip()
        entity_id  = self._desc_id(item_id)
        self._register_node(entity_id, NodeType.DESCRIPTION)
        return [Triplet(head=item_id, relation=Relation.HAS_DESCRIPTION, tail=entity_id)]

    def _extract_keywords(
        self,
        item_id: str,
        keywords: list[str],
    ) -> list[Triplet]:
        """Create one triplet per keyword."""
        triplets = []
        for kw in keywords:
            entity_id = self._keyword_id(kw)
            self._register_node(entity_id, NodeType.KEYWORD)
            triplets.append(
                Triplet(head=item_id, relation=Relation.HAS_KEYWORD, tail=entity_id)
            )
        return triplets

    # -------------------------------------------------------------------
    # Node id helpers  (centralised so naming is consistent)
    # -------------------------------------------------------------------

    @staticmethod
    def _item_id(title: str) -> str:
        return f"item::{title.strip()}"

    @staticmethod
    def _category_id(value: str) -> str:
        return f"cat::{value.strip()}"

    @staticmethod
    def _desc_id(item_id: str) -> str:
        return f"desc::{item_id}"

    @staticmethod
    def _keyword_id(kw: str) -> str:
        return f"kw::{kw.strip().lower()}"

    def _register_node(self, node_id: str, node_type: str) -> None:
        if node_id not in self._node_types:
            self._node_types[node_id] = node_type