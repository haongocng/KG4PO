from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.embedding.node_embedder import EmbeddingStore
from src.graph.builder import CKGraph
from .attention_scorer import AttentionScorer


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PropagationConfig:
    """
    Hyperparameters for the embedding propagation loop.

    Attributes
    ----------
    num_layers       : number of propagation layers L  (KGAT paper uses 3)
    aggregator       : 'bi_interaction' | 'gcn' | 'graphsage'
    hidden_dim       : output dimension of each aggregation layer.
                       None = keep same as input dim (no projection).
    dropout          : dropout rate applied to aggregated vectors [0, 1)
    temperature      : softmax temperature for attention scorer
    device           : 'cuda' | 'cpu'  (used for torch ops if available)
    """
    num_layers:   int   = 3
    aggregator:   str   = "bi_interaction"
    hidden_dim:   int | None = None
    dropout:      float = 0.1
    temperature:  float = 1.0
    device:       str   = "cuda"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PropagationResult:
    """
    Output of one full propagation run.

    Attributes
    ----------
    layer_embeddings  : list of length (num_layers + 1) —
                        layer_embeddings[l][node_id] = float32 vector at layer l.
                        layer_embeddings[0] is the initial EmbeddingStore vectors.
    attention_records : nested dict —
                        attention_records[layer][head_id] =
                            list of (relation, tail_id, weight)
    final_embeddings  : concatenated representation e* for each node —
                        e*[node_id] = concat(e^(0), e^(1), ..., e^(L))
    """
    layer_embeddings:  list[dict[str, np.ndarray]]
    attention_records: dict[int, dict[str, list[tuple[str, str, float]]]]
    final_embeddings:  dict[str, np.ndarray]

    def get_final(self, node_id: str) -> np.ndarray | None:
        """Return the concatenated final embedding for a node."""
        return self.final_embeddings.get(node_id)

    def get_attention(
        self,
        layer:   int,
        head_id: str,
    ) -> list[tuple[str, str, float]]:
        """
        Return attention weights for a specific (layer, head_id) pair.

        Returns:
            list of (relation, tail_id, weight) sorted by descending weight,
            or empty list if not found.
        """
        return self.attention_records.get(layer, {}).get(head_id, [])

    def top_k_attended(
        self,
        node_id: str,
        k:       int = 5,
    ) -> list[tuple[int, str, str, float]]:
        """
        Aggregate attention weights across all layers for a given node and
        return the top-k (layer, relation, tail_id, weight) entries.

        Useful for building the knowledge_retrieve text in the retrieval step.

        Args:
            node_id : head node to query
            k       : number of top entries to return

        Returns:
            list of (layer, relation, tail_id, weight) sorted by descending weight
        """
        all_entries: list[tuple[int, str, str, float]] = []
        for layer, head_map in self.attention_records.items():
            for rel, tail, w in head_map.get(node_id, []):
                all_entries.append((layer, rel, tail, w))

        all_entries.sort(key=lambda x: x[3], reverse=True)
        return all_entries[:k]


# ---------------------------------------------------------------------------
# PropagationLayer  —  main class
# ---------------------------------------------------------------------------

class PropagationLayer:
    """
    Runs L rounds of attentive embedding propagation over the CKGraph.

    Each round:
      1. For every node h, compute attention-weighted ego-network vector e_Nh
      2. Aggregate e_h and e_Nh using the chosen aggregator
      3. Store the updated embeddings for the next round

    After L rounds, concatenate all layer outputs into a final embedding e*.

    Aggregators
    -----------
    bi_interaction (default, best in KGAT paper):
        e^(l) = LeakyReLU(W1 (e_h + e_Nh)) + LeakyReLU(W2 (e_h ⊙ e_Nh))

    gcn:
        e^(l) = LeakyReLU(W (e_h + e_Nh))

    graphsage:
        e^(l) = LeakyReLU(W [e_h || e_Nh])
    """

    def __init__(
        self,
        graph:  CKGraph,
        store:  EmbeddingStore,
        config: PropagationConfig | None = None,
    ):
        """
        Args:
            graph  : fully built CKGraph
            store  : base EmbeddingStore (layer-0 vectors)
            config : propagation hyperparameters
        """
        self.graph  = graph
        self.store  = store
        self.cfg    = config or PropagationConfig()

        self._dim   = store.dim
        self._out_dim = self.cfg.hidden_dim or self._dim

        # Initialise weight matrices for aggregation (W1, W2)
        # Shape: (out_dim, in_dim) — applied as W @ v
        rng = np.random.default_rng(seed=0)
        scale = 1.0 / np.sqrt(self._dim)

        if self.cfg.aggregator == "graphsage":
            # GraphSage concatenates e_h || e_Nh so input dim is 2 * dim
            self._W1 = (rng.standard_normal((self._out_dim, self._dim * 2)) * scale
                        ).astype(np.float32)
            self._W2 = None
        else:
            # GCN and Bi-Interaction both take e_h + e_Nh (same dim)
            self._W1 = (rng.standard_normal((self._out_dim, self._dim)) * scale
                        ).astype(np.float32)
            self._W2 = (rng.standard_normal((self._out_dim, self._dim)) * scale
                        ).astype(np.float32)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def run(self, target_nodes: list[str] | None = None) -> PropagationResult:
        """
        Execute L layers of attentive embedding propagation.

        Args:
            target_nodes : if given, only propagate for these head nodes.
                           All nodes are still used as neighbors.
                           None = propagate for every node in the graph.

        Returns:
            PropagationResult with layer_embeddings, attention_records,
            and final_embeddings
        """
        nodes_to_propagate = (
            target_nodes if target_nodes is not None
            else list(self.graph.all_nodes)
        )

        # Layer 0: initial embeddings from EmbeddingStore
        current_embeddings: dict[str, np.ndarray] = self._init_layer_zero()

        layer_embeddings:  list[dict[str, np.ndarray]]              = [current_embeddings]
        attention_records: dict[int, dict[str, list[tuple[str, str, float]]]] = {}

        for layer_idx in range(1, self.cfg.num_layers + 1):
            scorer = AttentionScorer(
                store=self.store,
                graph=self.graph,
                temperature=self.cfg.temperature,
            )

            new_embeddings: dict[str, np.ndarray] = {}
            layer_attention: dict[str, list[tuple[str, str, float]]] = {}

            for head_id in nodes_to_propagate:
                e_h = current_embeddings.get(head_id)
                if e_h is None:
                    continue

                # Step 1: compute attention-weighted neighbor vector
                scored = scorer.score_neighbors(
                    head_id,
                    embeddings=current_embeddings,
                )
                layer_attention[head_id] = scored

                e_Nh = self._aggregate_neighbors(scored, current_embeddings)

                # Step 2: aggregate e_h and e_Nh
                e_new = self._aggregate(e_h, e_Nh)

                # Step 3: apply dropout
                e_new = self._dropout(e_new)

                new_embeddings[head_id] = e_new

            # Carry over nodes that were not in target_nodes
            # (needed so neighbor lookups in next layer still work)
            for nid, vec in current_embeddings.items():
                if nid not in new_embeddings:
                    new_embeddings[nid] = vec

            layer_embeddings.append(new_embeddings)
            attention_records[layer_idx] = layer_attention
            current_embeddings = new_embeddings

        # Build final concatenated embeddings e* = e^(0) || e^(1) || ... || e^(L)
        final_embeddings = self._concat_layers(layer_embeddings, nodes_to_propagate)

        return PropagationResult(
            layer_embeddings=layer_embeddings,
            attention_records=attention_records,
            final_embeddings=final_embeddings,
        )

    # -------------------------------------------------------------------
    # Aggregation helpers
    # -------------------------------------------------------------------

    def _aggregate_neighbors(
        self,
        scored:      list[tuple[str, str, float]],
        embeddings:  dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Compute e_Nh = sum_t pi(h,r,t) * e_t

        Returns a zero vector if no neighbor embeddings are available.
        """
        e_Nh = np.zeros(self._dim, dtype=np.float32)
        for _rel, tail_id, weight in scored:
            e_t = embeddings.get(tail_id)
            if e_t is None:
                e_t = self.store.get(tail_id)
            if e_t is not None:
                e_Nh += weight * e_t
        return e_Nh

    def _aggregate(self, e_h: np.ndarray, e_Nh: np.ndarray) -> np.ndarray:
        """
        Apply the configured aggregator to produce the new node embedding.

        bi_interaction:
            LeakyReLU(W1 @ (e_h + e_Nh)) + LeakyReLU(W2 @ (e_h * e_Nh))

        gcn:
            LeakyReLU(W1 @ (e_h + e_Nh))

        graphsage:
            LeakyReLU(W1 @ [e_h || e_Nh])
        """
        agg = self.cfg.aggregator

        if agg == "bi_interaction":
            part1 = self._leaky_relu(self._W1 @ (e_h + e_Nh))
            part2 = self._leaky_relu(self._W2 @ (e_h * e_Nh))   # type: ignore[operator]
            return part1 + part2

        elif agg == "gcn":
            return self._leaky_relu(self._W1 @ (e_h + e_Nh))

        elif agg == "graphsage":
            concat = np.concatenate([e_h, e_Nh], axis=0)
            return self._leaky_relu(self._W1 @ concat)

        else:
            raise ValueError(f"Unknown aggregator: {agg!r}. "
                             "Choose from 'bi_interaction', 'gcn', 'graphsage'.")

    # -------------------------------------------------------------------
    # Layer initialisation and finalisation
    # -------------------------------------------------------------------

    def _init_layer_zero(self) -> dict[str, np.ndarray]:
        """
        Build the layer-0 embedding dict from the EmbeddingStore.
        Every node in the graph gets a vector (zero if not in store).
        """
        embeddings: dict[str, np.ndarray] = {}
        zero = np.zeros(self._dim, dtype=np.float32)
        for node_id in self.graph.all_nodes:
            vec = self.store.get(node_id)
            embeddings[node_id] = vec if vec is not None else zero.copy()
        return embeddings

    def _concat_layers(
        self,
        layer_embeddings: list[dict[str, np.ndarray]],
        node_ids:         list[str],
    ) -> dict[str, np.ndarray]:
        """
        Concatenate embeddings across all layers for each node:
            e*[node] = e^(0)[node] || e^(1)[node] || ... || e^(L)[node]
        """
        final: dict[str, np.ndarray] = {}
        zero_base = np.zeros(self._dim, dtype=np.float32)
        zero_out  = np.zeros(self._out_dim, dtype=np.float32)

        for node_id in node_ids:
            parts: list[np.ndarray] = []
            for l_idx, l_emb in enumerate(layer_embeddings):
                vec = l_emb.get(node_id)
                if vec is None:
                    # Use appropriate zero shape per layer
                    vec = zero_base if l_idx == 0 else zero_out.copy()
                parts.append(vec)
            final[node_id] = np.concatenate(parts, axis=0)

        return final

    # -------------------------------------------------------------------
    # Math helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _leaky_relu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        return np.where(x >= 0, x, alpha * x).astype(np.float32)

    def _dropout(self, x: np.ndarray) -> np.ndarray:
        """
        Apply inverted dropout during propagation.
        Rate comes from config; no-op when rate == 0.
        """
        rate = self.cfg.dropout
        if rate <= 0.0:
            return x
        mask = (np.random.rand(*x.shape) > rate).astype(np.float32)
        return x * mask / (1.0 - rate)