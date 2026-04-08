from dataclasses import dataclass

from src.data.loader import SessionSample
from .triplet_extractor import Triplet, Relation, NodeType


@dataclass
class CoOccurConfig:
    """
    Configuration for co-occurrence edge generation.

    Attributes:
        window_size : max distance between two items in a session to be
                      considered co-occurring.  None = all pairs (full session).
        min_session_length : sessions shorter than this are skipped.
        max_session_length : sessions longer than this are truncated from the
                             end before processing (keeps most-recent items).
        use_position_weight: if True, store position info so downstream
                             attention can discount distant co-occurrences.
    """
    window_size:        int | None = 5
    min_session_length: int        = 1
    max_session_length: int        = 50
    use_position_weight: bool      = True


class CoOccurHandler:
    """
    Build co-occurrence triplets from session data.

    Strategy
    --------
    Instead of directly connecting every item pair (O(n^2) edges), we
    introduce a SESSION NODE as an intermediary:

        session_node  --co_occur-->  item_A
        session_node  --co_occur-->  item_B
        ...

    This keeps the number of edges linear in session length O(n) while
    still encoding the shared-session signal.  Two items that appear in
    the same session are then connected via a 2-hop path through the
    session node, which KGAT's multi-layer propagation naturally exploits.

    If window_size is set, only items within `window_size` positions of
    each other contribute to the same "window session node", creating
    finer-grained locality signals.
    """

    SESSION_PREFIX = "session"

    def __init__(self, config: CoOccurConfig | None = None):
        self.cfg = config or CoOccurConfig()
        self._node_types: dict[str, str] = {}

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def extract(self, sample: SessionSample) -> list[Triplet]:
        """
        Extract co-occurrence triplets for a single session sample.

        Args:
            sample: one SessionSample (contains session_items list)

        Returns:
            list of Triplet objects  (session_node --co_occur--> item)
        """
        items = self._prepare_items(sample.session_items)
        if len(items) < self.cfg.min_session_length:
            return []

        if self.cfg.window_size is None:
            return self._full_session_triplets(sample, items)
        else:
            return self._windowed_triplets(sample, items)

    def extract_all(self, samples: list[SessionSample]) -> list[Triplet]:
        """
        Extract co-occurrence triplets for all session samples.

        Deduplicates across samples so identical (session_node, item) pairs
        from different samples are not repeated.

        Args:
            samples: list of SessionSample objects

        Returns:
            deduplicated list of Triplet objects
        """
        seen:     set[Triplet]  = set()
        triplets: list[Triplet] = []

        for sample in samples:
            for t in self.extract(sample):
                if t not in seen:
                    seen.add(t)
                    triplets.append(t)

        return triplets

    def get_node_types(self) -> dict[str, str]:
        """Return node_type registry built during extraction."""
        return dict(self._node_types)

    # -------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------

    def _prepare_items(self, items: list[str]) -> list[str]:
        """
        Clean and truncate the item list according to config.
        Keeps the LAST max_session_length items (most recent interactions).
        """
        cleaned = [i.strip() for i in items if i.strip()]
        if len(cleaned) > self.cfg.max_session_length:
            cleaned = cleaned[-self.cfg.max_session_length :]
        return cleaned

    def _full_session_triplets(
        self,
        sample: SessionSample,
        items: list[str],
    ) -> list[Triplet]:
        """
        One session node per sample; all items connect to it.
        Used when window_size is None.
        """
        session_id = self._session_node_id(sample, window_idx=None)
        self._register_node(session_id, NodeType.SESSION)

        triplets: list[Triplet] = []
        for item in items:
            item_id = self._item_node_id(item)
            self._register_node(item_id, NodeType.ITEM)
            triplets.append(
                Triplet(head=session_id, relation=Relation.CO_OCCUR, tail=item_id)
            )
        return triplets

    def _windowed_triplets(
        self,
        sample: SessionSample,
        items: list[str],
    ) -> list[Triplet]:
        """
        Sliding window approach: for each position i, create a window session
        node that covers items[i : i + window_size].

        This produces finer locality signals — items close in sequence share
        a window node, items far apart do not.

        Window session node id encodes both the sample target and window index
        so different windows remain distinct in the graph.
        """
        w         = self.cfg.window_size          # type: ignore[assignment]
        triplets: list[Triplet] = []
        seen_pairs: set[tuple[str, str]] = set()

        for start in range(0, len(items), w):      # non-overlapping windows
            window_items = items[start : start + w]
            win_idx      = start // w
            session_id   = self._session_node_id(sample, window_idx=win_idx)
            self._register_node(session_id, NodeType.SESSION)

            for item in window_items:
                item_id = self._item_node_id(item)
                pair    = (session_id, item_id)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                self._register_node(item_id, NodeType.ITEM)
                triplets.append(
                    Triplet(head=session_id, relation=Relation.CO_OCCUR, tail=item_id)
                )

        return triplets

    # -------------------------------------------------------------------
    # Node id helpers
    # -------------------------------------------------------------------

    def _session_node_id(
        self,
        sample: SessionSample,
        window_idx: int | None,
    ) -> str:
        """
        Unique session node identifier.
        Format:
          full   window : session::<target>::full
          sliding window: session::<target>::w<idx>
        """
        suffix = "full" if window_idx is None else f"w{window_idx}"
        # Use target title as the session anchor (unique per sample)
        safe_target = sample.target.replace(" ", "_").replace("/", "-")
        return f"{self.SESSION_PREFIX}::{safe_target}::{suffix}"

    @staticmethod
    def _item_node_id(title: str) -> str:
        return f"item::{title.strip()}"

    def _register_node(self, node_id: str, node_type: str) -> None:
        if node_id not in self._node_types:
            self._node_types[node_id] = node_type