from enum import Enum
from typing import Any


class SchemaType(Enum):
    TAXONOMY = "taxonomy"       # Co truong taxonomy phan cap Level_1..Level_N
    CATEGORY = "category"       # Co truong category phang (set/list nhan)
    UNKNOWN  = "unknown"        # Khong xac dinh duoc


class SchemaDetector:
    """
    Tu dong phat hien kieu du lieu cua product info.

    Ho tro hai dang:
      - TAXONOMY : { "taxonomy": { "Level_1": ..., "Level_2": ..., ... } }
      - CATEGORY : { "category": [...] hoac {...} }
    """

    # Cac ten truong taxonomy hop le
    TAXONOMY_FIELD = "taxonomy"
    CATEGORY_FIELD = "category"
    CATEGORIES_FIELD = "categories"
    LEVEL_PREFIX   = "Level_"

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    @classmethod
    def detect(cls, product: dict[str, Any]) -> SchemaType:
        """
        Nhan vao mot product dict, tra ve SchemaType tuong ung.

        Args:
            product: mot phan tu trong danh sach product info

        Returns:
            SchemaType.TAXONOMY | SchemaType.CATEGORY | SchemaType.UNKNOWN
        """
        if cls._is_taxonomy(product):
            return SchemaType.TAXONOMY
        if cls._is_category(product):
            return SchemaType.CATEGORY
        return SchemaType.UNKNOWN

    @classmethod
    def detect_corpus(cls, products: list[dict[str, Any]]) -> SchemaType:
        """
        Phat hien schema cho toan bo corpus bang cach lay majority vote.
        Huu ich khi mot so phan tu bi thieu truong.

        Args:
            products: danh sach tat ca product info

        Returns:
            SchemaType chiem da so
        """
        if not products:
            return SchemaType.UNKNOWN

        counts: dict[SchemaType, int] = {
            SchemaType.TAXONOMY: 0,
            SchemaType.CATEGORY: 0,
            SchemaType.UNKNOWN:  0,
        }
        for p in products:
            counts[cls.detect(p)] += 1

        return max(counts, key=lambda k: counts[k])

    @classmethod
    def extract_taxonomy_levels(
        cls, product: dict[str, Any]
    ) -> dict[str, str]:
        """
        Trich xuat cac cap bac taxonomy thanh dict { "Level_1": value, ... }.
        Chi goi khi detect() tra ve TAXONOMY.

        Returns:
            dict sap xep theo thu tu Level (Level_1 truoc)
        """
        raw: dict = product.get(cls.TAXONOMY_FIELD, {})
        levels = {
            k: v
            for k, v in raw.items()
            if k.startswith(cls.LEVEL_PREFIX) and isinstance(v, str) and v.strip()
        }
        # Sap xep Level_1, Level_2, ...
        return dict(
            sorted(levels.items(), key=lambda kv: cls._level_order(kv[0]))
        )

    # @classmethod
    # def extract_categories(
    #     cls, product: dict[str, Any]
    # ) -> list[str]:
    #     """
    #     Trich xuat danh sach category phang.
    #     Chi goi khi detect() tra ve CATEGORY.

    #     Returns:
    #         list cac category string, da loai bo gia tri rong
    #     """
    #     raw = product.get(cls.CATEGORY_FIELD, [])

    #     if isinstance(raw, dict):
    #         # Truong hop { "0": "Action", "1": "Drama" }
    #         values = list(raw.values())
    #     elif isinstance(raw, (list, set, tuple)):
    #         values = list(raw)
    #     elif isinstance(raw, str):
    #         values = [raw]
    #     else:
    #         values = []

    #     return [str(v).strip() for v in values if str(v).strip()]

    @classmethod
    def extract_categories(
        cls, product: dict[str, Any]
    ) -> list[str]:
        """
        Trich xuat danh sach category phang.
        Ho tro ca truong "category" (so it) lan "categories" (so nhieu).
        Chi goi khi detect() tra ve CATEGORY.
        """
        # Prefer singular "category" first, fall back to plural "categories"
        raw = product.get(cls.CATEGORY_FIELD) \
              or product.get(cls.CATEGORIES_FIELD, [])
 
        if isinstance(raw, dict):
            values = list(raw.values())
        elif isinstance(raw, (list, set, tuple)):
            values = list(raw)
        elif isinstance(raw, str):
            values = [raw]
        else:
            values = []
 
        return [str(v).strip() for v in values if str(v).strip()]

    # -------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------

    @classmethod
    def _is_taxonomy(cls, product: dict[str, Any]) -> bool:
        taxonomy = product.get(cls.TAXONOMY_FIELD)
        if not isinstance(taxonomy, dict):
            return False
        # Phai co it nhat 1 truong Level_x
        return any(k.startswith(cls.LEVEL_PREFIX) for k in taxonomy)

    @classmethod
    def _is_category(cls, product: dict[str, Any]) -> bool:
        return cls.CATEGORY_FIELD in product

    @staticmethod
    def _level_order(level_key: str) -> int:
        """Tra ve so thu tu cua Level_x de sap xep."""
        try:
            return int(level_key.split("_")[1])
        except (IndexError, ValueError):
            return 999