from __future__ import annotations

import numpy as np

from src.embedding.node_embedder import EmbeddingStore
from src.graph.builder import CKGraph


class AttentionScorer:
    """
    Compute knowledge-aware attention scores pi(h, r, t) for every
    (head, relation, tail) triplet in the ego-network of a node.

    Follows the KGAT attention formulation:
        pi(h, r, t) = (W_r * e_t)^T  tanh(W_r * e_h + e_r)

    Since we use pre-trained text embeddings (no learned W_r per relation),
    we approximate the relation-space projection with a lightweight
    relation-aware dot-product:

        score(h, r, t) = e_t^T  tanh(e_h + e_r)

    where e_r is the mean embedding of all tail nodes connected to h
    via relation r (a simple relational context vector derived from the
    training data without requiring learned parameters).

    If a relation has no pre-computed context vector, we fall back to
    plain cosine similarity between e_h and e_t.

    All scores for a given head node are normalised with softmax so they
    sum to 1.0 across the full ego-network.
    """

    def __init__(
        self,
        store:  EmbeddingStore,
        graph:  CKGraph,
        temperature: float = 1.0,
    ):
        """
        Args:
            store       : EmbeddingStore with pre-computed node vectors
            graph       : CKGraph (used to derive relation context vectors)
            temperature : softmax temperature — lower = sharper distribution
        """
        self.store        = store
        self.graph        = graph
        self.temperature  = temperature

        # Pre-compute relation context vectors: rel -> mean tail embedding
        self._rel_context: dict[str, np.ndarray] = self._build_relation_context()

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def score_neighbors(
        self,
        head_id: str,
        embeddings: dict[str, np.ndarray] | None = None,
    ) -> list[tuple[str, str, float]]:
        """
        Compute softmax-normalised attention weights for all neighbors
        of head_id.

        Args:
            head_id    : node whose ego-network we score
            embeddings : optional override dict (node_id -> vector) —
                         used during propagation when node vectors have
                         been updated in a previous layer

        Returns:
            list of (relation, tail_id, attention_weight) sorted by
            descending attention weight
        """
        neighbors = self.graph.neighbors(head_id)
        if not neighbors:
            return []

        e_h = self._get_vec(head_id, embeddings)
        if e_h is None:
            # Head not in store — uniform attention
            w = 1.0 / len(neighbors)
            return [(r, t, w) for r, t in neighbors]

        raw_scores: list[float] = []
        for rel, tail_id in neighbors:
            e_t = self._get_vec(tail_id, embeddings)
            if e_t is None:
                raw_scores.append(0.0)
            else:
                raw_scores.append(self._compute_score(e_h, rel, e_t))

        weights = self._softmax(np.array(raw_scores, dtype=np.float32))

        result = [
            (rel, tail_id, float(w))
            for (rel, tail_id), w in zip(neighbors, weights)
        ]
        # Sort descending by attention weight for easy top-k extraction later
        result.sort(key=lambda x: x[2], reverse=True)
        return result

    def score_node_list(
        self,
        node_ids: list[str],
        embeddings: dict[str, np.ndarray] | None = None,
    ) -> dict[str, list[tuple[str, str, float]]]:
        """
        Batch version: compute attention scores for a list of head nodes.

        Args:
            node_ids   : list of head node ids to score
            embeddings : optional embedding override dict

        Returns:
            dict mapping head_id -> list of (relation, tail_id, weight)
        """
        return {
            nid: self.score_neighbors(nid, embeddings=embeddings)
            for nid in node_ids
        }

    # -------------------------------------------------------------------
    # Score computation
    # -------------------------------------------------------------------

    def _compute_score(
        self,
        e_h:  np.ndarray,
        rel:  str,
        e_t:  np.ndarray,
    ) -> float:
        """
        Relation-aware attention score for a single (h, r, t) triple.

        Formula:
            context  = tanh(e_h + e_r)          if e_r exists
                     = tanh(e_h)                 fallback
            score    = dot(e_t, context)
        """
        e_r = self._rel_context.get(rel)
        if e_r is not None:
            context = np.tanh(e_h + e_r)
        else:
            context = np.tanh(e_h)

        return float(np.dot(e_t, context))

    # -------------------------------------------------------------------
    # Relation context vectors
    # -------------------------------------------------------------------

    def _build_relation_context(self) -> dict[str, np.ndarray]:
        """
        For each relation type in the graph, compute a context vector as
        the L2-normalised mean of all tail node embeddings connected via
        that relation.

        This approximates the relation embedding e_r without requiring
        explicit relation-specific weight matrices (which would need training).
        """
        rel_accum:  dict[str, list[np.ndarray]] = {}

        for triplet in self.graph.triplets:
            vec = self.store.get(triplet.tail)
            if vec is None:
                continue
            if triplet.relation not in rel_accum:
                rel_accum[triplet.relation] = []
            rel_accum[triplet.relation].append(vec)

        rel_context: dict[str, np.ndarray] = {}
        for rel, vecs in rel_accum.items():
            mean_vec = np.mean(vecs, axis=0).astype(np.float32)
            norm     = np.linalg.norm(mean_vec)
            if norm > 1e-8:
                mean_vec = mean_vec / norm
            rel_context[rel] = mean_vec

        return rel_context

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _get_vec(
        self,
        node_id:    str,
        embeddings: dict[str, np.ndarray] | None,
    ) -> np.ndarray | None:
        """
        Look up a node vector, preferring the override dict (current-layer
        embeddings) over the base EmbeddingStore.
        """
        if embeddings is not None and node_id in embeddings:
            return embeddings[node_id]
        return self.store.get(node_id)

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        """
        Numerically stable softmax with temperature scaling.

        Args:
            scores: raw score array

        Returns:
            probability array summing to 1.0
        """
        scaled = scores / max(self.temperature, 1e-8)
        shifted = scaled - scaled.max()          # numerical stability
        exp_s   = np.exp(shifted)
        return exp_s / (exp_s.sum() + 1e-10)