import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema_detector import SchemaDetector, SchemaType


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SessionSample:
    """Mot mau du lieu tu train/test.json."""
    target: str                         # San pham can du doan
    target_index: int                   # Vi tri target trong candidate set
    session_items: list[str]            # Danh sach san pham trong phien (theo thu tu)
    candidate_items: list[str]          # Danh sach ung vien de chon
    raw_input: str = ""                 # Chuoi input goc (de debug)


@dataclass
class ProductInfo:
    """Thong tin mot san pham tu info.json."""
    title: str
    schema_type: SchemaType
    taxonomy_levels: dict[str, str]     # Chi co khi TAXONOMY
    categories: list[str]               # Chi co khi CATEGORY
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


@dataclass
class LoadedData:
    """Ket qua tong hop sau khi load xong ca hai file."""
    samples: list[SessionSample]
    products: dict[str, ProductInfo]    # key = title (normalized)
    corpus_schema: SchemaType


# ---------------------------------------------------------------------------
# DataLoader
# ---------------------------------------------------------------------------

class DataLoader:
    """
    Load va parse:
      - train.json / test.json  →  list[SessionSample]
      - info_*.json             →  dict[str, ProductInfo]
    """

    # Regex tach danh sach san pham trong chuoi input
    _ITEM_PATTERN = re.compile(r'\d+\."([^"]+)"')

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def load(
        self,
        session_path: str | Path,
        info_path: str | Path,
    ) -> LoadedData:
        """
        Load ca hai file va tra ve LoadedData.

        Args:
            session_path : duong dan toi train.json hoac test.json
            info_path    : duong dan toi info_*.json

        Returns:
            LoadedData chua samples, products va corpus_schema
        """
        raw_sessions  = self._read_json(session_path)
        raw_products  = self._read_json(info_path)

        products      = self._parse_products(raw_products)
        corpus_schema = SchemaDetector.detect_corpus(raw_products)
        samples       = self._parse_sessions(raw_sessions)

        return LoadedData(
            samples=samples,
            products=products,
            corpus_schema=corpus_schema,
        )

    def load_sessions(self, path: str | Path) -> list[SessionSample]:
        """Chi load file session (train/test)."""
        return self._parse_sessions(self._read_json(path))

    def load_products(
        self, path: str | Path
    ) -> tuple[dict[str, ProductInfo], SchemaType]:
        """Chi load file product info."""
        raw      = self._read_json(path)
        products = self._parse_products(raw)
        schema   = SchemaDetector.detect_corpus(raw)
        return products, schema

    # -------------------------------------------------------------------
    # Parse sessions
    # -------------------------------------------------------------------

    def _parse_sessions(
        self, raw: list[dict[str, Any]]
    ) -> list[SessionSample]:
        samples = []
        for entry in raw:
            try:
                sample = self._parse_one_session(entry)
                samples.append(sample)
            except Exception as exc:
                target = entry.get("target", "?")
                print(f"[DataLoader] Warning: bo qua sample '{target}': {exc}")
        return samples

    def _parse_one_session(self, entry: dict[str, Any]) -> SessionSample:
        target       = str(entry["target"]).strip()
        target_index = int(entry.get("target_index", -1))
        raw_input    = str(entry.get("input", ""))

        session_items, candidate_items = self._parse_input_string(raw_input)

        return SessionSample(
            target=target,
            target_index=target_index,
            session_items=session_items,
            candidate_items=candidate_items,
            raw_input=raw_input,
        )

    def _parse_input_string(
        self, raw_input: str
    ) -> tuple[list[str], list[str]]:
        """
        Tach chuoi input thanh (session_items, candidate_items).

        Dinh dang:
          "Current session interactions: [1."A", 2."B", ...]
           Candidate Set: [1."X", 2."Y", ...]"
        """
        # Tach phan session va phan candidate
        parts = re.split(r"Candidate Set\s*:", raw_input, maxsplit=1)

        session_part   = parts[0] if len(parts) >= 1 else ""
        candidate_part = parts[1] if len(parts) == 2 else ""

        session_items   = self._ITEM_PATTERN.findall(session_part)
        candidate_items = self._ITEM_PATTERN.findall(candidate_part)

        return session_items, candidate_items

    # -------------------------------------------------------------------
    # Parse products
    # -------------------------------------------------------------------

    def _parse_products(
        self, raw: list[dict[str, Any]]
    ) -> dict[str, ProductInfo]:
        products: dict[str, ProductInfo] = {}
        for entry in raw:
            try:
                info = self._parse_one_product(entry)
                key  = self._normalize_title(info.title)
                products[key] = info
            except Exception as exc:
                title = entry.get("title", "?")
                print(f"[DataLoader] Warning: bo qua product '{title}': {exc}")
        return products

    def _parse_one_product(self, entry: dict[str, Any]) -> ProductInfo:
        title       = str(entry.get("title", "")).strip()
        schema_type = SchemaDetector.detect(entry)

        taxonomy_levels: dict[str, str] = {}
        categories: list[str]           = []

        if schema_type == SchemaType.TAXONOMY:
            taxonomy_levels = SchemaDetector.extract_taxonomy_levels(entry)
        elif schema_type == SchemaType.CATEGORY:
            categories = SchemaDetector.extract_categories(entry)

        # Lay description tu nhieu vi tri co the co
        details     = entry.get("details", {})
        description = (
            details.get("description", "")
            or entry.get("description", "")
        ).strip()

        # Lay keywords neu co
        keywords_raw = (
            details.get("keywords", [])
            or entry.get("keywords", [])
        )
        if isinstance(keywords_raw, str):
            keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
        elif isinstance(keywords_raw, (list, tuple)):
            keywords = [str(k).strip() for k in keywords_raw if str(k).strip()]
        else:
            keywords = []

        return ProductInfo(
            title=title,
            schema_type=schema_type,
            taxonomy_levels=taxonomy_levels,
            categories=categories,
            description=description,
            keywords=keywords,
            raw=entry,
        )

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _read_json(self, path: str | Path) -> list[dict[str, Any]]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found file: {path}")
        with open(path, encoding=self.encoding) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"File {path} must be JSON array, get {type(data)}")
        return data

    @staticmethod
    def _normalize_title(title: str) -> str:
        """Chuan hoa title de dung lam key tra cuu (lowercase, strip)."""
        return title.strip().lower()