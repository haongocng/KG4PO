from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import yaml

from src.data.loader import DataLoader, LoadedData
from src.graph.builder import CKGraph, GraphBuilder
from src.graph.co_occur_handler import CoOccurConfig
from src.embedding.node_embedder import EmbeddingStore, NodeEmbedder
from src.attention.propagation import PropagationConfig, PropagationLayer, PropagationResult
from src.retrieval.topk_selector import TopKSelector, TopKResult
from src.retrieval.text_synthesizer import KnowledgeEntry, TextSynthesizer


class Pipeline:
    """
    End-to-end orchestrator for the KGAT Knowledge Retrieve system.

    Flow
    ----
    1.  Load session data + product info           (DataLoader)
    2.  Build / load cached CKGraph                (GraphBuilder)
    3.  Compute / load cached node embeddings      (NodeEmbedder)
    4.  Run attentive embedding propagation        (PropagationLayer)
    5.  Select top-K attended neighbours           (TopKSelector)
    6.  Synthesise retrieved_knowledge per session (TextSynthesizer)
    7.  Write output JSONL

    Output format (one JSON object per line):
    {
        "target":             str,
        "target_index":       int,
        "input":              str,   -- original raw input string preserved
        "retrieved_knowledge": str   -- graph-inferred context for LLM
    }
    """

    def __init__(self, config: dict[str, Any]):
        self.cfg = config
        self._data_cfg  = config.get("data", {})
        self._graph_cfg = config.get("graph", {})
        self._emb_cfg   = config.get("embedding", {})
        self._prop_cfg  = config.get("propagation", {})
        self._ret_cfg   = config.get("retrieval", {})
        self._pipe_cfg  = config.get("pipeline", {})

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path = "config.yaml") -> "Pipeline":
        with open(path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return cls(config)

    def run(
        self,
        session_path: str | Path | None = None,
        info_path:    str | Path | None = None,
        output_path:  str | Path | None = None,
    ) -> list[dict]:
        t0 = time.perf_counter()

        s_path = Path(session_path or self._data_cfg.get("session_path", ""))
        p_path = Path(info_path    or self._data_cfg.get("info_path",    ""))
        o_path = Path(output_path  or self._data_cfg.get("output_path",
                      "data/processed/knowledge_retrieve.jsonl"))

        print(f"\n{'='*60}")
        print(f"  KGAT Knowledge Retrieve Pipeline")
        print(f"  session : {s_path}")
        print(f"  info    : {p_path}")
        print(f"  output  : {o_path}")
        print(f"{'='*60}\n")

        data   = self._step_load(s_path, p_path)
        graph  = self._step_graph(data)
        store  = self._step_embed(graph)
        result = self._step_propagate(graph, store, data)
        output = self._step_retrieve(data, graph, result)
        self._write_output(output, o_path)

        elapsed = time.perf_counter() - t0
        print(f"\n[Pipeline] Done — {len(output)} records written to {o_path}  "
              f"({elapsed:.1f}s total)\n")
        return output

    # -------------------------------------------------------------------
    # Pipeline steps
    # -------------------------------------------------------------------

    def _step_load(self, session_path: Path, info_path: Path) -> LoadedData:
        t = time.perf_counter()
        print("[1/6] Loading data ...")
        data = DataLoader().load(session_path, info_path)
        print(f"      {len(data.samples)} sessions, "
              f"{len(data.products)} products, "
              f"schema={data.corpus_schema.value}  "
              f"({time.perf_counter()-t:.1f}s)")
        return data

    def _step_graph(self, data: LoadedData) -> CKGraph:
        t          = time.perf_counter()
        cache_path = self._resolve_cache("graph_cache")

        if cache_path and cache_path.exists() and self._pipe_cfg.get("use_cache", True):
            print(f"[2/6] Loading graph from cache: {cache_path}")
            graph = CKGraph.load(cache_path)
        else:
            print("[2/6] Building CKGraph ...")
            co_cfg = CoOccurConfig(
                window_size=self._graph_cfg.get("co_occur_window_size", 5),
                min_session_length=self._graph_cfg.get("min_session_length", 1),
                max_session_length=self._graph_cfg.get("max_session_length", 50),
            )
            graph = GraphBuilder(
                co_occur_config=co_cfg,
                include_description=self._graph_cfg.get("include_description", True),
                include_keywords=self._graph_cfg.get("include_keywords", True),
            ).build(data)
            if cache_path:
                graph.save(cache_path)
                print(f"      Graph cached to {cache_path}")

        stats = graph.stats()
        print(f"      nodes={stats['num_nodes']}, triplets={stats['num_triplets']}, "
              f"relations={stats['num_relations']}  ({time.perf_counter()-t:.1f}s)")
        return graph

    def _step_embed(self, graph: CKGraph) -> EmbeddingStore:
        t          = time.perf_counter()
        cache_path = self._resolve_cache("embedding_cache")

        if cache_path and cache_path.exists() and self._pipe_cfg.get("use_cache", True):
            print(f"[3/6] Loading embeddings from cache: {cache_path}")
            store = EmbeddingStore.load(cache_path)
        else:
            print(f"[3/6] Computing embeddings  "
                  f"(model={self._emb_cfg.get('model_name', 'all-MiniLM-L6-v2')}, "
                  f"device={self._emb_cfg.get('device', 'cuda')}) ...")
            embedder = NodeEmbedder(
                model_name=self._emb_cfg.get("model_name", "all-MiniLM-L6-v2"),
                batch_size=self._emb_cfg.get("batch_size", 256),
                use_type_prefix=self._emb_cfg.get("use_type_prefix", True),
                device=self._emb_cfg.get("device", "cuda"),
                normalize=self._emb_cfg.get("normalize", True),
            )
            store = embedder.embed_graph(graph)
            if cache_path:
                store.save(cache_path)
                print(f"      Embeddings cached to {cache_path}")

        print(f"      {len(store)} node vectors, dim={store.dim}  "
              f"({time.perf_counter()-t:.1f}s)")
        return store

    def _step_propagate(
        self,
        graph: CKGraph,
        store: EmbeddingStore,
        data:  LoadedData,
    ) -> PropagationResult:
        t = time.perf_counter()
        print(f"[4/6] Running propagation  "
              f"(L={self._prop_cfg.get('num_layers', 3)}, "
              f"agg={self._prop_cfg.get('aggregator', 'bi_interaction')}) ...")

        # Propagate for all item nodes present in the graph
        # (not just session items, so bridge-node inference has full coverage)
        target_nodes = list(graph.item_nodes)

        prop_cfg = PropagationConfig(
            num_layers=self._prop_cfg.get("num_layers", 3),
            aggregator=self._prop_cfg.get("aggregator", "bi_interaction"),
            hidden_dim=self._prop_cfg.get("hidden_dim", None),
            dropout=self._prop_cfg.get("dropout", 0.1),
            temperature=self._prop_cfg.get("temperature", 1.0),
            device=self._emb_cfg.get("device", "cuda"),
        )
        result = PropagationLayer(
            graph=graph,
            store=store,
            config=prop_cfg,
        ).run(target_nodes=target_nodes)

        print(f"      {len(target_nodes)} item nodes propagated  "
              f"({time.perf_counter()-t:.1f}s)")
        return result

    def _step_retrieve(
        self,
        data:   LoadedData,
        graph:  CKGraph,
        result: PropagationResult,
    ) -> list[dict]:
        t = time.perf_counter()
        print("[5/6] Selecting top-K and synthesising retrieved_knowledge ...")

        synthesizer = TextSynthesizer(graph=graph, products=data.products)
        log_every   = self._pipe_cfg.get("log_every", 100)
        output: list[dict] = []

        for idx, sample in enumerate(data.samples):
            if log_every > 0 and idx > 0 and idx % log_every == 0:
                print(f"      processed {idx}/{len(data.samples)} sessions ...")

            # Build session-scoped selector so inferred items exclude session items
            session_item_ids = {f"item::{t}" for t in sample.session_items}
            selector = TopKSelector(
                graph=graph,
                k_categories=self._ret_cfg.get("k_categories", 4),
                k_keywords=self._ret_cfg.get("k_keywords", 3),
                k_inferred_items=self._ret_cfg.get("k_inferred_items", 3),
                k_descriptions=self._ret_cfg.get("k_descriptions", 1),
                layer_decay=self._ret_cfg.get("layer_decay", 0.9),
                exclude_self=self._ret_cfg.get("exclude_self", True),
                session_item_ids=session_item_ids,
            )

            record = self._process_one_session(
                sample=sample,
                graph=graph,
                result=result,
                selector=selector,
                synthesizer=synthesizer,
            )
            output.append(record)

        print(f"      {len(output)} sessions processed  "
              f"({time.perf_counter()-t:.1f}s)")
        return output

    # -------------------------------------------------------------------
    # Per-session processing
    # -------------------------------------------------------------------

    def _process_one_session(
        self,
        sample,
        graph:       CKGraph,
        result:      PropagationResult,
        selector:    TopKSelector,
        synthesizer: TextSynthesizer,
    ) -> dict:
        """
        Build one output record per session.

        Output schema:
        {
            "target":              str,
            "target_index":        int,
            "input":               str,   -- original raw input preserved
            "retrieved_knowledge": str    -- graph-inferred context for LLM
        }
        """
        # Collect top-K results for every item in the session
        topk_results: dict[str, TopKResult] = {}
        for title in sample.session_items:
            item_id = f"item::{title}"
            if item_id in graph.all_nodes:
                topk_results[item_id] = selector.select(item_id, result)

        # Build the single retrieved_knowledge string for this session
        retrieved_knowledge = synthesizer.build_session_knowledge(
            session_titles=sample.session_items,
            topk_results=topk_results,
        )

        return {
            "target":              sample.target,
            "target_index":        sample.target_index,
            "input":               sample.raw_input,
            "retrieved_knowledge": retrieved_knowledge,
        }

    # -------------------------------------------------------------------
    # Output  —  JSONL format (one JSON object per line)
    # -------------------------------------------------------------------

    def _write_output(self, output: list[dict], path: Path) -> None:
        print(f"[6/6] Writing output to {path} ...")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in output:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _resolve_cache(self, key: str) -> Path | None:
        raw = self._data_cfg.get(key)
        if not raw:
            return None
        return Path(raw)