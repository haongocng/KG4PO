from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.graph.builder import CKGraph
from src.graph.triplet_extractor import NodeType


# ---------------------------------------------------------------------------
# EmbeddingStore  —  holds all node vectors + lookup helpers
# ---------------------------------------------------------------------------

@dataclass
class EmbeddingStore:
    """
    Stores a float32 embedding matrix and a bidirectional index
    mapping node_id <-> row index.

    Attributes
    ----------
    matrix    : numpy array of shape (N, dim) — one row per node
    node2idx  : node_id  -> row index in matrix
    idx2node  : row index -> node_id
    dim       : embedding dimension
    """
    matrix:   np.ndarray
    node2idx: dict[str, int]
    idx2node: dict[int, str]
    dim:      int

    # -------------------------------------------------------------------
    # Lookup
    # -------------------------------------------------------------------

    def get(self, node_id: str) -> np.ndarray | None:
        """
        Return the embedding vector for a node_id, or None if not found.

        Args:
            node_id: graph node identifier

        Returns:
            float32 numpy array of shape (dim,) or None
        """
        idx = self.node2idx.get(node_id)
        if idx is None:
            return None
        return self.matrix[idx]

    def get_batch(self, node_ids: list[str]) -> np.ndarray:
        """
        Return a (len(node_ids), dim) matrix for a list of node_ids.
        Missing nodes are replaced with zero vectors.

        Args:
            node_ids: list of graph node identifiers

        Returns:
            float32 numpy array of shape (len(node_ids), dim)
        """
        out = np.zeros((len(node_ids), self.dim), dtype=np.float32)
        for i, nid in enumerate(node_ids):
            vec = self.get(nid)
            if vec is not None:
                out[i] = vec
        return out

    def __len__(self) -> int:
        return len(self.node2idx)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self.node2idx

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialise the store to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingStore":
        """Deserialise a previously saved EmbeddingStore."""
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected EmbeddingStore, got {type(obj)}")
        return obj


# ---------------------------------------------------------------------------
# NodeEmbedder
# ---------------------------------------------------------------------------

class NodeEmbedder:
    """
    Compute text embeddings for every node in a CKGraph using a
    sentence-transformer model.

    Node text is constructed per node type:
      - ITEM        : the item title                   e.g. "Aliens"
      - CATEGORY    : the category label               e.g. "Military Sci-Fi"
      - SESSION     : a short descriptor               e.g. "user session"
      - DESCRIPTION : the description text of the item
      - KEYWORD     : the keyword string

    The text for each node is derived by stripping the type prefix from
    the node_id (e.g. "item::Aliens" -> "Aliens") and then optionally
    enriching it with a type context prefix that helps the encoder
    understand the semantic role of each node.
    """

    # Prefix injected before the raw node text to provide context to the encoder
    _TYPE_PREFIX: dict[str, str] = {
        NodeType.ITEM:        "Title:",
        NodeType.CATEGORY:    "Category:",
        NodeType.SESSION:     "Viewing session context",
        NodeType.DESCRIPTION: "Description:",
        NodeType.KEYWORD:     "Keyword:",
    }

    # Separator between the type prefix ("item::") and the actual value
    _ID_SEPARATOR = "::"

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        use_type_prefix: bool = True,
        device: str | None = None,
        normalize: bool = True,
    ):
        """
        Args:
            model_name      : sentence-transformers model name or local path
            batch_size      : number of texts to encode per forward pass
            use_type_prefix : prepend semantic role prefix to each node text
            device          : 'cpu', 'cuda', or None (auto-detect)
            normalize       : L2-normalise embeddings (recommended for cosine similarity)
        """
        self.model_name      = model_name
        self.batch_size      = batch_size
        self.use_type_prefix = use_type_prefix
        self.device          = device
        self.normalize       = normalize
        self._model          = None          # Lazy load on first call

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def embed_graph(self, graph: CKGraph) -> EmbeddingStore:
        """
        Compute embeddings for all nodes in the graph.

        Args:
            graph: fully built CKGraph

        Returns:
            EmbeddingStore with one vector per node
        """
        node_ids = sorted(graph.all_nodes)          # deterministic order
        texts    = [
            self._node_to_text(nid, graph.node_type(nid))
            for nid in node_ids
        ]

        print(f"[NodeEmbedder] Encoding {len(node_ids)} nodes "
              f"with model='{self.model_name}' ...")

        matrix = self._encode(texts)                # (N, dim)

        node2idx = {nid: i for i, nid in enumerate(node_ids)}
        idx2node = {i: nid for i, nid in enumerate(node_ids)}

        store = EmbeddingStore(
            matrix=matrix,
            node2idx=node2idx,
            idx2node=idx2node,
            dim=matrix.shape[1],
        )
        print(f"[NodeEmbedder] Done — matrix shape: {matrix.shape}")
        return store

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Encode an arbitrary list of raw text strings.
        Useful for encoding query items at inference time.

        Args:
            texts: list of strings to encode

        Returns:
            float32 numpy array of shape (len(texts), dim)
        """
        return self._encode(texts)

    # -------------------------------------------------------------------
    # Node text construction
    # -------------------------------------------------------------------

    def _node_to_text(self, node_id: str, node_type: str) -> str:
        """
        Convert a node_id + node_type into a text string for the encoder.

        Strategy:
          1. Strip the type prefix from node_id  (e.g. "item::Aliens" -> "Aliens")
          2. Optionally prepend a semantic role prefix for context
        """
        raw_text = self._strip_prefix(node_id)

        if not self.use_type_prefix:
            return raw_text

        role_prefix = self._TYPE_PREFIX.get(node_type, "")

        # Session nodes carry no meaningful text — use a fixed descriptor
        if node_type == NodeType.SESSION:
            return role_prefix

        return f"{role_prefix} {raw_text}".strip()

    def _strip_prefix(self, node_id: str) -> str:
        """
        Remove the type prefix from a node_id.

        Examples:
          "item::Aliens"            -> "Aliens"
          "cat::Military Sci-Fi"    -> "Military Sci-Fi"
          "desc::item::Aliens"      -> "item::Aliens"   (one level stripped)
          "kw::action"              -> "action"
          "session::Aliens::w0"     -> "Aliens::w0"
        """
        sep = self._ID_SEPARATOR
        if sep in node_id:
            return node_id[node_id.index(sep) + len(sep):]
        return node_id

    # -------------------------------------------------------------------
    # Encoding
    # -------------------------------------------------------------------

    def _encode(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of texts in batches.

        Returns float32 numpy array of shape (len(texts), dim).
        Falls back to a zero-vector stub if sentence-transformers is not
        installed, so the rest of the pipeline can be tested without GPU deps.
        """
        model = self._get_model()

        if model is None:
            # Stub: return random unit vectors for testing without the library
            print("[NodeEmbedder] WARNING: sentence-transformers not available. "
                  "Using random embeddings (for testing only).")
            dim    = 384
            rng    = np.random.default_rng(seed=42)
            matrix = rng.standard_normal((len(texts), dim)).astype(np.float32)
            if self.normalize:
                norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
                matrix = matrix / np.clip(norms, 1e-8, None)
            return matrix

        # Encode in batches to avoid OOM on large graphs
        all_vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vecs  = model.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
            )
            all_vectors.append(vecs.astype(np.float32))

        return np.vstack(all_vectors)

    # -------------------------------------------------------------------
    # Model loading (lazy)
    # -------------------------------------------------------------------

    def _get_model(self):
        """
        Lazy-load the sentence-transformers model on first call.
        Returns None if the library is not installed.
        """
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer
            kwargs: dict = {"model_name_or_path": self.model_name}
            if self.device is not None:
                kwargs["device"] = self.device
            self._model = SentenceTransformer(**kwargs)
            print(f"[NodeEmbedder] Loaded model: {self.model_name}")
        except ImportError:
            self._model = None

        return self._model