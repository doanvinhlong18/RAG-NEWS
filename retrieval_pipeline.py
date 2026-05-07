"""
retrieval_pipeline.py
=====================
Full multi-stage hybrid retrieval pipeline:

  Query → [Expansion] → BM25 + Dense → [RRF] → [Bi-Encoder Rerank] → [CE Rerank] → Top-K

Stages:
  1. Query Processing:   rule-based rewrite + multi-query expansion (2–3 variants)
  2. Hybrid Retrieval:   BM25 (CPU) + FAISS (CPU) → top-100 each
  3. RRF Fusion:         Reciprocal Rank Fusion → merged top-100
  4. Bi-Encoder Rerank:  Stage 1 — top-100 → 30 (semantic similarity)
  5. Cross-Encoder Rerank: Stage 2 — top-30 → 5 (fine-grained relevance)
     NOTE: No score fusion after CE (scores used only for ordering)

Design Decisions:
  - Lazy model loading (load only when needed)
  - GPU → CPU fallback on OOM
  - Embedding cache per query (in-memory LRU)
  - BM25 and FAISS results normalized before RRF (rank-based, not score-based)

Run:
  python retrieval_pipeline.py --query "Who won the 2016 US election?"
  python retrieval_pipeline.py --query "..." --config config.yaml
"""

import os
import re
import pickle
import logging
import argparse
from typing import Dict, List, Tuple

import yaml
import numpy as np

from multi_query_retriever import (
    QueryIntentClassifier,
    expand_query_with_intent,
    retrieve_multi_query,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 1. Query Processing
# ---------------------------------------------------------------------------
class QueryProcessor:
    """
    Intent-aware query rewriter and multi-query expander.

    Classifies the query intent (statement / biography / event / causal /
    opinion / general) then generates intent-specific expansion variants.
    This fixes known retrieval weaknesses:
      - Statement queries now include subject+topic keyword variants.
      - Biography queries now include background/role keyword variants.
    """

    def __init__(self, n_variants: int = 3, use_t5: bool = False):
        self.n_variants  = n_variants
        self.use_t5      = use_t5          # reserved — T5 rewriter not used
        self.classifier  = QueryIntentClassifier()
        self.last_intent = "general"       # set after each classify_and_expand call

    def rewrite(self, query: str) -> str:
        """Light cleaning: collapse whitespace."""
        return re.sub(r"\s+", " ", query).strip()

    def classify_and_expand(self, query: str) -> tuple:
        """
        Classify query intent and return intent-aware expansion variants.

        Returns:
            (intent: str, variants: List[str])
        """
        query = self.rewrite(query)
        intent = self.classifier.classify(query)
        self.last_intent = intent
        variants = expand_query_with_intent(query, intent, self.n_variants)
        logger.debug(f"Intent={intent} | variants={variants}")
        return intent, variants

    def expand(self, query: str) -> List[str]:
        """Backward-compatible expand — uses intent-aware expansion internally."""
        _, variants = self.classify_and_expand(query)
        return variants


# ---------------------------------------------------------------------------
# 2. Hybrid Retrieval (BM25 + Dense)
# ---------------------------------------------------------------------------
class HybridRetriever:
    """
    BM25 (CPU) + FAISS dense (CPU) retrieval with lazy loading.
    Scores are NOT fused directly — only ranks are used in RRF.
    """

    def __init__(
        self,
        bm25_path: str,
        faiss_path: str,
        meta_path: str,
        bi_encoder_model: str,
        corpus_path: str = "",
        max_seq_length: int = 256,
        top_k: int = 100,
        rrf_k: int = 60,
        use_gpu: bool = True,
        dense_query_batch_size: int = 32,
    ):
        self.bm25_path        = bm25_path
        self.faiss_path       = faiss_path
        self.meta_path        = meta_path
        self.corpus_path      = corpus_path
        self.bi_encoder_model = bi_encoder_model
        self.max_seq_length   = max_seq_length
        self.top_k            = top_k
        self.rrf_k            = rrf_k
        self.use_gpu          = use_gpu
        self.dense_query_batch_size = dense_query_batch_size

        self._bm25         = None
        self._faiss        = None
        self._metadata     = None
        self._encoder      = None
        self._device       = None
        self._stop_words   = None
        self._chunk_lookup = None  # chunk_id → {chunk_text, title}

    # ── Lazy Loaders ─────────────────────────────────────────────────────────

    def _get_chunk_lookup(self) -> dict:
        """Build chunk_id → {chunk_text, title} from corpus_chunks.jsonl (lazy).
        Caches result as a pickle next to the JSONL so subsequent runs skip JSON parsing.
        """
        if self._chunk_lookup is not None:
            return self._chunk_lookup

        import json

        if not self.corpus_path or not os.path.exists(self.corpus_path):
            logger.warning(f"corpus_chunks.jsonl not found at '{self.corpus_path}'. chunk_text will be empty.")
            self._chunk_lookup = {}
            return self._chunk_lookup

        cache_path = self.corpus_path.replace(".jsonl", "_lookup.pkl")

        if os.path.exists(cache_path):
            logger.info(f"Loading chunk lookup cache from {cache_path} …")
            with open(cache_path, "rb") as f:
                self._chunk_lookup = pickle.load(f)
            logger.info(f"Chunk lookup loaded: {len(self._chunk_lookup):,} entries")
            return self._chunk_lookup

        logger.info(f"Loading chunk text lookup from {self.corpus_path} …")
        lookup = {}
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                lookup[rec["chunk_id"]] = {
                    "chunk_text": rec.get("chunk_text", ""),
                    "title":      rec.get("title", ""),
                }

        logger.info(f"Saving chunk lookup cache to {cache_path} …")
        with open(cache_path, "wb") as f:
            pickle.dump(lookup, f)

        self._chunk_lookup = lookup
        logger.info(f"Chunk lookup loaded: {len(lookup):,} entries")
        return self._chunk_lookup

    def _get_bm25(self):
        if self._bm25 is None:
            logger.info(f"Loading BM25 index from {self.bm25_path} …")
            with open(self.bm25_path, "rb") as f:
                self._bm25 = pickle.load(f)
        return self._bm25

    def _get_faiss(self):
        if self._faiss is None:
            import faiss
            logger.info(f"Loading FAISS index from {self.faiss_path} …")
            self._faiss = faiss.read_index(self.faiss_path)
            logger.info(f"FAISS: {self._faiss.ntotal:,} vectors")
        return self._faiss

    def _get_metadata(self):
        if self._metadata is None:
            with open(self.meta_path, "rb") as f:
                self._metadata = pickle.load(f)
        return self._metadata

    def _get_encoder(self):
        if self._encoder is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cpu"
            if self.use_gpu and torch.cuda.is_available():
                free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                if free > 600 * 1024 ** 2:
                    device = "cuda"
            self._device = device
            logger.info(f"Loading bi-encoder: {self.bi_encoder_model} on {device} …")
            self._encoder = SentenceTransformer(self.bi_encoder_model, device=device)
            self._encoder.max_seq_length = self.max_seq_length
            if device == "cuda":
                self._encoder = self._encoder.half()
        return self._encoder

    def _get_stop_words(self):
        if self._stop_words is None:
            import nltk
            from nltk.corpus import stopwords
            self._stop_words = set(stopwords.words("english"))
        return self._stop_words

    # ── BM25 Search ──────────────────────────────────────────────────────────

    def _bm25_search(self, query: str) -> List[Tuple[str, float]]:
        """Returns list of (chunk_id, bm25_score) sorted desc."""
        from data_pipeline.utils import tokenize_for_bm25
        bm25     = self._get_bm25()
        metadata = self._get_metadata()
        stops    = self._get_stop_words()

        toks   = tokenize_for_bm25(query, stops)
        scores = bm25.get_scores(toks)
        top_n  = min(self.top_k, len(scores))
        top_ix = np.argpartition(scores, -top_n)[-top_n:]
        top_ix = top_ix[np.argsort(-scores[top_ix])]

        return [(metadata[i]["chunk_id"], float(scores[i])) for i in top_ix]

    # ── Dense Search ─────────────────────────────────────────────────────────

    def _dense_search(self, query: str) -> List[Tuple[str, float]]:
        """Returns list of (chunk_id, cosine_score) sorted desc."""
        return self._dense_search_batch([query], batch_size=1)[0]

    def _dense_search_batch(self, queries: List[str], batch_size: int) -> List[List[Tuple[str, float]]]:
        """Batch dense search. Returns list aligned with queries."""
        import torch
        encoder   = self._get_encoder()
        faiss_idx = self._get_faiss()
        metadata  = self._get_metadata()

        try:
            embs = encoder.encode(
                queries,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
            ).astype(np.float32)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self._device == "cuda":
                logger.warning("OOM during batch query encode — falling back to CPU.")
                torch.cuda.empty_cache()
                encoder = encoder.to("cpu").float()
                self._device = "cpu"
                embs = encoder.encode(
                    queries,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                ).astype(np.float32)
            else:
                raise

        D, I = faiss_idx.search(embs, self.top_k)
        results: List[List[Tuple[str, float]]] = []
        for j in range(len(queries)):
            results.append([
                (metadata[i]["chunk_id"], float(D[j][k]))
                for k, i in enumerate(I[j]) if i >= 0
            ])
        return results

    # ── RRF Fusion ───────────────────────────────────────────────────────────

    def _rrf_fusion(
        self,
        bm25_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """
        Reciprocal Rank Fusion.
        RRF score = 1/(k + rank_bm25) + 1/(k + rank_dense)
        NOTE: Uses only ranks, not raw scores — normalizes across retrieval methods.
        """
        k = self.rrf_k
        rrf: Dict[str, float] = {}

        for rank, (cid, _) in enumerate(bm25_results, start=1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank)

        for rank, (cid, _) in enumerate(dense_results, start=1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank)

        return sorted(rrf.items(), key=lambda x: -x[1])

    # ── Public: Retrieve ─────────────────────────────────────────────────────

    def retrieve(
        self,
        queries: List[str],
        return_full_chunks: bool = True,
    ) -> List[Dict]:
        """
        Hybrid retrieval for a list of query variants.

        Merges BM25 + dense results per query via RRF, then fuses across queries.
        Returns list of chunk dicts with 'rrf_score'.
        """
        metadata = self._get_metadata()

        dense_batch = self._dense_search_batch(queries, batch_size=self.dense_query_batch_size)
        fused = retrieve_multi_query(
            queries,
            bm25_search=self._bm25_search,
            dense_batch_results=dense_batch,
            top_k=self.top_k,
            rrf_k=self.rrf_k,
        )

        if not return_full_chunks:
            return [{"chunk_id": cid, "rrf_score": s} for cid, s in fused]

        # Enrich with full chunk data
        cid_to_meta    = {c["chunk_id"]: c for c in metadata.values()}
        chunk_lookup   = self._get_chunk_lookup()   # chunk_id → {chunk_text, title}
        results = []
        for cid, rrf_score in fused:
            chunk = dict(cid_to_meta.get(cid, {}))
            # chunk_metadata.pkl may not have chunk_text — fall back to corpus JSONL
            if not chunk.get("chunk_text") and cid in chunk_lookup:
                chunk.update(chunk_lookup[cid])
            results.append({**chunk, "rrf_score": rrf_score})

        return results


# ---------------------------------------------------------------------------
# 3. Bi-Encoder Reranker (Stage 1: 100 → 30)
# ---------------------------------------------------------------------------
class BiEncoderReranker:
    """
    Reranks candidates using semantic similarity from the bi-encoder.
    Stage 1: top-100 candidates → top-30.
    """

    def __init__(self, model_name: str, max_seq_length: int = 256, top_k: int = 30, use_gpu: bool = True):
        self.model_name     = model_name
        self.max_seq_length = max_seq_length
        self.top_k          = top_k
        self.use_gpu        = use_gpu
        self._model         = None

    def _get_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            device = "cpu"
            if self.use_gpu and torch.cuda.is_available():
                free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                if free > 600 * 1024 ** 2:
                    device = "cuda"
            logger.info(f"Loading bi-encoder reranker: {self.model_name} on {device} …")
            self._model = SentenceTransformer(self.model_name, device=device)
            self._model.max_seq_length = self.max_seq_length
            if device == "cuda":
                self._model = self._model.half()
        return self._model

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        Re-score candidates by cosine(query_emb, doc_emb).
        Returns top-k sorted by bi-encoder score.
        NOTE: bi_encoder_score is used ONLY for ordering here; CE will re-score next.
        """
        if not candidates:
            return candidates

        model = self._get_model()

        texts = [c.get("chunk_text", "") for c in candidates]
        q_emb = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        d_embs = model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)

        scores = (q_emb @ d_embs.T).flatten()

        ranked = sorted(
            zip(candidates, scores.tolist()),
            key=lambda x: -x[1],
        )[:self.top_k]

        for chunk, score in ranked:
            chunk["bi_encoder_score"] = score

        return [c for c, _ in ranked]


# ---------------------------------------------------------------------------
# 4. Cross-Encoder Reranker (Stage 2: 30 → 5)
# ---------------------------------------------------------------------------
class CrossEncoderReranker:
    """
    Reranks candidates using Cross-Encoder (full query-document interaction).
    Stage 2: top-30 → top-5.
    NOTE: CE scores are used for final ordering ONLY — not fused with any other scores.
    """

    def __init__(self, model_name: str, max_length: int = 256, top_k: int = 5, use_gpu: bool = True):
        self.model_name = model_name
        self.max_length = max_length
        self.top_k      = top_k
        self.use_gpu    = use_gpu
        self._model     = None

    def _get_model(self):
        if self._model is None:
            import torch
            from sentence_transformers.cross_encoder import CrossEncoder
            device = "cuda" if (self.use_gpu and torch.cuda.is_available()) else "cpu"
            logger.info(f"Loading cross-encoder: {self.model_name} on {device} …")
            self._model = CrossEncoder(self.model_name, max_length=self.max_length, device=device)
        return self._model

    def rerank(self, query: str, candidates: List[Dict], score_threshold: float = 0.0) -> List[Dict]:
        """
        Score all (query, doc) pairs. Return top-k above threshold.
        CE scores are NOT fused with previous-stage scores.
        """
        if not candidates:
            return candidates

        model  = self._get_model()
        pairs  = [(query, c.get("chunk_text", "")) for c in candidates]
        scores = model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip(candidates, scores.tolist()),
            key=lambda x: -x[1],
        )

        result = []
        for chunk, score in ranked:
            if score >= score_threshold:
                chunk["ce_score"] = score
                result.append(chunk)
            if len(result) >= self.top_k:
                break

        return result


# ---------------------------------------------------------------------------
# 5. Full Pipeline
# ---------------------------------------------------------------------------
class RetrievalPipeline:
    """
    Orchestrates the full multi-stage retrieval pipeline.

    Query → Expand → BM25+Dense → RRF → BiEncoder Rerank → CE Rerank → Top-K
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        ret_cfg  = self.cfg["retrieval"]

        data_dir  = self.cfg["paths"]["data_dir"]
        index_dir = self.cfg["paths"]["index_dir"]

        # Determine which bi-encoder model to use
        if ret_cfg["use_finetuned_bi_encoder"] and os.path.exists(self.cfg["models"]["bi_encoder_finetuned"]):
            bi_model = self.cfg["models"]["bi_encoder_finetuned"]
        else:
            bi_model = self.cfg["models"]["bi_encoder_base"]

        if ret_cfg["use_finetuned_cross_encoder"] and os.path.exists(self.cfg["models"]["cross_encoder_finetuned"]):
            ce_model = self.cfg["models"]["cross_encoder_finetuned"]
        else:
            ce_model = self.cfg["models"]["cross_encoder_base"]

        # Components
        qe_cfg = ret_cfg["query_expansion"]
        self.query_proc = QueryProcessor(
            n_variants=qe_cfg["n_variants"],
            use_t5=qe_cfg["use_t5_rewriter"],
        )

        self.hybrid = HybridRetriever(
            bm25_path   = os.path.join(index_dir, "bm25.pkl"),
            faiss_path  = os.path.join(index_dir, "faiss.index"),
            meta_path   = os.path.join(index_dir, "chunk_metadata.pkl"),
            corpus_path = os.path.join(data_dir, "corpus_chunks.jsonl"),
            bi_encoder_model = bi_model,
            max_seq_length   = self.cfg["bi_encoder_training"]["max_seq_length"],
            top_k  = ret_cfg["bm25_top_k"],
            rrf_k  = ret_cfg["rrf_k"],
            dense_query_batch_size = ret_cfg.get("dense_query_batch_size", 32),
        )

        self.bi_reranker = BiEncoderReranker(
            model_name    = bi_model,
            max_seq_length = self.cfg["bi_encoder_training"]["max_seq_length"],
            top_k = ret_cfg["bi_encoder_rerank_top_k"],
        )

        self.ce_reranker = CrossEncoderReranker(
            model_name  = ce_model,
            max_length  = self.cfg["cross_encoder_training"]["max_seq_length"],
            top_k       = ret_cfg["cross_encoder_rerank_top_k"],
        )

        self.score_threshold = ret_cfg["ce_score_threshold"]

    def retrieve(self, query: str, verbose: bool = False) -> List[Dict]:
        """Full pipeline: query → top-K reranked documents."""
        # Stage 0: Intent classification + query expansion
        if self.cfg["retrieval"]["query_expansion"]["enabled"]:
            intent, queries = self.query_proc.classify_and_expand(query)
        else:
            intent = "general"
            queries = [query]
            self.query_proc.last_intent = intent

        if verbose:
            logger.info(f"Intent: {intent} | Query variants: {queries}")

        # Stage 1: Hybrid retrieval (BM25 + Dense → RRF)
        candidates = self.hybrid.retrieve(queries)
        if verbose:
            logger.info(f"After hybrid retrieval: {len(candidates)} candidates")

        # Stage 2: Bi-Encoder rerank (100 → 30)
        candidates = self.bi_reranker.rerank(query, candidates)
        if verbose:
            logger.info(f"After bi-encoder rerank: {len(candidates)} candidates")

        # Stage 3: Cross-Encoder rerank (30 → 5) — NO score fusion, ordering only
        final = self.ce_reranker.rerank(query, candidates, self.score_threshold)
        if verbose:
            logger.info(f"After cross-encoder rerank: {len(final)} results")

        return final


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Multi-Stage Hybrid Retrieval Pipeline")
    parser.add_argument("--query",  required=True, help="Query string")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    pipeline = RetrievalPipeline(args.config)
    results  = pipeline.retrieve(args.query, verbose=args.verbose)

    print(f"\n{'='*70}")
    print(f"Query: {args.query}")
    print(f"{'='*70}")
    for i, doc in enumerate(results, 1):
        ce_score  = doc.get("ce_score", "n/a")
        rrf_score = doc.get("rrf_score", 0.0)
        print(f"\n[{i}] doc_id={doc.get('doc_id', '?')} | CE={ce_score:.4f} | RRF={rrf_score:.4f}")
        print(f"    Title : {doc.get('title', '')[:80]}")
        print(f"    Text  : {doc.get('chunk_text', '')[:200]} …")
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
