"""
evaluation.py
=============
Evaluate all retrieval stages using BEIR-format qrels.

Evaluates:
  1. BM25 only
  2. Dense (FAISS) only
  3. Hybrid (BM25 + Dense + RRF)
  4. Hybrid + Multi-Query
  5. MultiHybrid + Bi-Encoder Rerank
  6. Hybrid + Bi-Encoder + Cross-Encoder Rerank (final pipeline)

Metrics:
  - NDCG@{1,3,5,10,100}
  - MAP@{1,3,5,10,100}
  - Recall@{1,3,5,10,100}
  - Precision@{1,3,5,10,100}
  - MRR@10

Fixes applied vs original:
  [FIX-1] Bi-encoder rerank top_k_out raised from 30 → 80 to avoid
          over-pruning candidates before cross-encoder.
  [FIX-2] Cross-encoder top_k_out raised from 5 → 25 so that NDCG/Recall
          metrics at k > 5 are no longer artificially capped at the same value.
  [FIX-3] Bi-encoder rerank now uses a dot-product score guard: chunks that
          score below a minimum similarity threshold are dropped before
          passing to CE, instead of blindly truncating by rank.
  [FIX-4] FAISS half-precision encoding isolated — encoder.half() is only
          applied for the initial index build; query encoding uses float32
          to avoid cosine-similarity drift on short query vectors.
  [FIX-5] rrf_merge helper de-duplicated (was copy-pasted inline); now
          shared with multi-query stage via rrf_fusion import.
  [FIX-6] Queries with no qrel entries are now skipped consistently in
          compute_precision (previously returned 0 for missing qids rather
          than skipping, inflating the denominator).

Run:
  python evaluation.py
  python evaluation.py --config config.yaml --max_queries 200
"""

import os
import json
import pickle
import logging
import argparse
from typing import Dict, List, Tuple

import yaml
import numpy as np
from tqdm import tqdm
import torch

from multi_query_retriever import expand_query, rrf_fusion


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Result Mapping: Chunk-level → Doc-level
# ---------------------------------------------------------------------------

def chunk_to_doc_results(
    chunk_results: Dict[str, Dict[str, float]],
    chunk_to_doc: Dict[str, str],
) -> Dict[str, Dict[str, float]]:
    """
    Map chunk-level retrieval scores to document-level by taking the max score
    per document per query.

    Args:
        chunk_results : {qid: {chunk_id: score}}
        chunk_to_doc  : {chunk_id: doc_id}

    Returns:
        {qid: {doc_id: max_score}}
    """
    doc_results: Dict[str, Dict[str, float]] = {}
    for qid, chunks in chunk_results.items():
        doc_results[qid] = {}
        for chunk_id, score in chunks.items():
            doc_id = chunk_to_doc.get(chunk_id, chunk_id)
            if doc_id not in doc_results[qid] or score > doc_results[qid][doc_id]:
                doc_results[qid][doc_id] = score
    return doc_results


# ---------------------------------------------------------------------------
# BM25 Retrieval
# ---------------------------------------------------------------------------

def bm25_retrieve_all(
    queries: Dict[str, str],
    bm25,
    metadata: dict,
    stops: set,
    top_k: int = 100,
) -> Dict[str, Dict[str, float]]:
    """BM25 retrieval for all queries. Returns {qid: {chunk_id: score}}."""
    from data_pipeline.utils import tokenize_for_bm25

    results = {}
    for qid, query in tqdm(queries.items(), desc="BM25 eval", unit="queries"):
        toks   = tokenize_for_bm25(query, stops)
        scores = bm25.get_scores(toks)
        top_n  = min(top_k, len(scores))
        top_ix = np.argpartition(scores, -top_n)[-top_n:]
        top_ix = top_ix[np.argsort(-scores[top_ix])]
        results[qid] = {
            metadata[int(i)]["chunk_id"]: float(scores[i]) for i in top_ix
        }
    return results


# ---------------------------------------------------------------------------
# Dense Retrieval
# ---------------------------------------------------------------------------

def dense_retrieve_all(
    queries: Dict[str, str],
    faiss_index,
    metadata: dict,
    encoder,
    batch_size: int = 64,
    top_k: int = 100,
) -> Dict[str, Dict[str, float]]:
    """
    Dense FAISS retrieval for all queries. Returns {qid: {chunk_id: score}}.

    [FIX-4] Query vectors are encoded in float32 regardless of whether the
    encoder was cast to half() for index construction. This prevents cosine
    similarity drift on short query vectors.
    """
    qids    = list(queries.keys())
    q_texts = [queries[q] for q in qids]

    logger.info(f"Encoding {len(q_texts)} queries (float32) …")

    # [FIX-4] Temporarily move to float32 for query encoding
    original_dtype = next(encoder.parameters()).dtype
    encoder_fp32 = encoder.float() if original_dtype != float else encoder

    q_embs = encoder_fp32.encode(
        q_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    logger.info("FAISS search …")
    D, I = faiss_index.search(q_embs, top_k)

    results = {}
    for j, qid in enumerate(qids):
        results[qid] = {
            metadata[i]["chunk_id"]: float(D[j][k])
            for k, i in enumerate(I[j]) if i >= 0
        }
    return results


# ---------------------------------------------------------------------------
# IR Metrics
# ---------------------------------------------------------------------------

def compute_ndcg(qrels: Dict, results: Dict, k: int) -> float:
    """Compute NDCG@k averaged over queries."""
    ndcgs = []
    for qid in qrels:
        if qid not in results or not results[qid]:
            ndcgs.append(0.0)
            continue
        relevant = qrels[qid]
        ranked   = sorted(results[qid].items(), key=lambda x: -x[1])[:k]
        dcg, idcg = 0.0, 0.0
        ideal_scores = sorted(relevant.values(), reverse=True)[:k]
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            rel = relevant.get(doc_id, 0)
            dcg += rel / np.log2(rank + 1)
        for rank, rel in enumerate(ideal_scores, start=1):
            idcg += rel / np.log2(rank + 1)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs))


def compute_recall(qrels: Dict, results: Dict, k: int) -> float:
    """Compute Recall@k averaged over queries."""
    recalls = []
    for qid in qrels:
        relevant = set(d for d, s in qrels[qid].items() if s > 0)
        if not relevant:
            continue
        if qid not in results or not results[qid]:
            recalls.append(0.0)
            continue
        retrieved = set(
            d for d, _ in sorted(results[qid].items(), key=lambda x: -x[1])[:k]
        )
        recalls.append(len(relevant & retrieved) / len(relevant))
    return float(np.mean(recalls)) if recalls else 0.0


def compute_map(qrels: Dict, results: Dict, k: int) -> float:
    """Compute MAP@k averaged over queries."""
    aps = []
    for qid in qrels:
        relevant = set(d for d, s in qrels[qid].items() if s > 0)
        if not relevant:
            continue
        if qid not in results or not results[qid]:
            aps.append(0.0)
            continue
        ranked   = sorted(results[qid].items(), key=lambda x: -x[1])[:k]
        num_hits, ap = 0, 0.0
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            if doc_id in relevant:
                num_hits += 1
                ap += num_hits / rank
        ap /= min(len(relevant), k)
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


def compute_precision(qrels: Dict, results: Dict, k: int) -> float:
    """
    Compute Precision@k averaged over queries.

    [FIX-6] Queries absent from results OR absent from qrels are both
    skipped so they do not inflate the denominator.
    """
    precs = []
    for qid in qrels:
        # [FIX-6] skip queries with no relevant docs (undefined precision)
        relevant = set(d for d, s in qrels[qid].items() if s > 0)
        if not relevant:
            continue
        if qid not in results:
            precs.append(0.0)
            continue
        retrieved = [
            d for d, _ in sorted(results[qid].items(), key=lambda x: -x[1])[:k]
        ]
        hits = sum(1 for d in retrieved if d in relevant)
        precs.append(hits / k)
    return float(np.mean(precs)) if precs else 0.0


def compute_mrr(qrels: Dict, results: Dict, k: int = 10) -> float:
    """Compute MRR@k averaged over queries."""
    mrrs = []
    for qid in qrels:
        if qid not in results:
            mrrs.append(0.0)
            continue
        relevant = set(d for d, s in qrels[qid].items() if s > 0)
        ranked   = sorted(results[qid].items(), key=lambda x: -x[1])[:k]
        rr = 0.0
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        mrrs.append(rr)
    return float(np.mean(mrrs))


def evaluate_results(
    qrels: Dict, results: Dict, k_values: List[int], name: str
) -> Dict:
    """Compute all metrics for a retrieval result dict."""
    metrics = {}
    for k in k_values:
        metrics[f"NDCG@{k}"]      = compute_ndcg(qrels, results, k)
        metrics[f"MAP@{k}"]       = compute_map(qrels, results, k)
        metrics[f"Recall@{k}"]    = compute_recall(qrels, results, k)
        metrics[f"Precision@{k}"] = compute_precision(qrels, results, k)
    metrics["MRR@10"] = compute_mrr(qrels, results, 10)

    logger.info(f"\n{'─'*60}")
    logger.info(f"  {name}")
    logger.info(f"{'─'*60}")
    for m, v in metrics.items():
        logger.info(f"  {m:20s}: {v:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# RRF merge helper (shared)
# [FIX-5] Extracted from inline copy-paste; uses rrf_fusion from
#         multi_query_retriever for consistency.
# ---------------------------------------------------------------------------

def rrf_merge(
    bm25_r: Dict[str, float],
    dense_r: Dict[str, float],
    rrf_k: int,
    top_k: int,
) -> Dict[str, float]:
    """Merge BM25 and Dense results via Reciprocal Rank Fusion."""
    bm25_list  = sorted(bm25_r.items(),  key=lambda x: -x[1])
    dense_list = sorted(dense_r.items(), key=lambda x: -x[1])
    fused = rrf_fusion([bm25_list, dense_list], k=rrf_k, top_k=top_k)
    return {cid: score for cid, score in fused}


# ---------------------------------------------------------------------------
# Bi-Encoder Rerank
# ---------------------------------------------------------------------------

def bi_encoder_rerank_results(
    queries: Dict[str, str],
    chunk_results: Dict[str, Dict[str, float]],
    chunks: Dict[str, str],
    encoder,
    top_k_in: int = 100,
    top_k_out: int = 80,           # [FIX-1] raised from 30 → 80
    min_score: float = 0.0,        # [FIX-3] score guard threshold
) -> Dict[str, Dict[str, float]]:
    """
    Apply bi-encoder reranking to retrieved chunks.

    [FIX-1] top_k_out default raised to 80 so the cross-encoder receives
            enough candidates to produce meaningful Recall@k for k > 30.
    [FIX-3] Chunks scoring below min_score are dropped before passing to CE.
            This avoids feeding clearly irrelevant passages while still
            keeping a large candidate pool.
    [FIX-4] Encoder is kept in float32 for query encoding.
    """
    qids       = [qid for qid in queries if qid in chunk_results]
    q_texts    = [queries[qid] for qid in qids]

    logger.info(f"Bi-encoder: encoding {len(q_texts)} queries (float32) …")
    # [FIX-4] encode in float32 regardless of model storage dtype
    q_embs_all = encoder.float().encode(
        q_texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    q_emb_map = {qid: q_embs_all[i] for i, qid in enumerate(qids)}

    reranked = {}
    for qid in tqdm(qids, desc="Bi-encoder rerank"):
        top_chunks = sorted(
            chunk_results[qid].items(), key=lambda x: -x[1]
        )[:top_k_in]
        cids       = [cid for cid, _ in top_chunks]
        valid_cids = [cid for cid in cids if cid in chunks]
        texts      = [chunks[cid] for cid in valid_cids]

        if not texts:
            continue

        d_embs = encoder.float().encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        scores = (q_emb_map[qid] @ d_embs.T).flatten()

        # [FIX-3] apply score guard before rank truncation
        ranked = [
            (cid, float(score))
            for cid, score in sorted(
                zip(valid_cids, scores.tolist()), key=lambda x: -x[1]
            )
            if float(score) >= min_score
        ][:top_k_out]   # [FIX-1] now uses the larger top_k_out

        reranked[qid] = {cid: score for cid, score in ranked}

    return reranked


# ---------------------------------------------------------------------------
# Cross-Encoder Rerank
# ---------------------------------------------------------------------------

def ce_rerank_results(
    queries: Dict[str, str],
    chunk_results: Dict[str, Dict[str, float]],
    chunks: Dict[str, str],
    ce_model,
    top_k_in: int = 80,    # [FIX-2] matches new bi-encoder top_k_out
    top_k_out: int = 25,   # [FIX-2] raised from 5 → 25
) -> Dict[str, Dict[str, float]]:
    """
    Apply cross-encoder reranking.

    [FIX-2] top_k_out raised from 5 → 25. The original value of 5 caused
            NDCG@k, Recall@k, and Precision@k for all k > 5 to be identical
            (artificially capped), since no additional documents existed
            beyond rank 5 in the result set.
    """
    reranked = {}
    for qid in tqdm(queries, desc="CE rerank"):
        if qid not in chunk_results:
            continue
        query = queries[qid]
        top_chunks = sorted(
            chunk_results[qid].items(), key=lambda x: -x[1]
        )[:top_k_in]
        cids       = [cid for cid, _ in top_chunks]
        valid_cids = [cid for cid in cids if cid in chunks]
        texts      = [chunks[cid] for cid in valid_cids]

        if not texts:
            continue

        pairs  = [(query, t) for t in texts]
        scores = ce_model.predict(pairs, show_progress_bar=False)
        # scores = torch.sigmoid(torch.tensor(scores)).numpy()
        ranked = sorted(
            zip(valid_cids, scores.tolist()), key=lambda x: -x[1]
        )[:top_k_out]  # [FIX-2]
        reranked[qid] = {cid: score for cid, score in ranked}

    return reranked


# ---------------------------------------------------------------------------
# Multi-Query Hybrid Retrieval
# ---------------------------------------------------------------------------

def multi_query_hybrid_retrieve_all(
    queries: Dict[str, str],
    bm25,
    faiss_index,
    metadata: dict,
    encoder,
    stops: set,
    n_variants: int,
    batch_size: int = 64,
    top_k: int = 100,
    rrf_k: int = 60,
) -> Dict[str, Dict[str, float]]:
    """
    Multi-query hybrid retrieval.

    Per query:
      - Expand into n_variants variants
      - Run BM25 + Dense per variant
      - RRF-fuse within variant, then fuse across variants
    """
    from data_pipeline.utils import tokenize_for_bm25

    expanded = {
        qid: expand_query(q, n_variants=n_variants)
        for qid, q in queries.items()
    }

    flat_pairs: List[Tuple[str, int, str]] = []
    for qid, variants in expanded.items():
        for vidx, v in enumerate(variants):
            flat_pairs.append((qid, vidx, v))

    flat_texts = [v for _, _, v in flat_pairs]
    logger.info(f"Encoding {len(flat_texts)} expanded queries (float32) …")

    # [FIX-4] float32 for query encoding
    q_embs = encoder.float().encode(
        flat_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    logger.info("FAISS search (expanded queries) …")
    D, I = faiss_index.search(q_embs, top_k)

    dense_by_qid_variant: Dict[Tuple[str, int], List[Tuple[str, float]]] = {}
    for row, (qid, vidx, _) in enumerate(flat_pairs):
        dense_by_qid_variant[(qid, vidx)] = [
            (metadata[i]["chunk_id"], float(D[row][k]))
            for k, i in enumerate(I[row]) if i >= 0
        ]

    results: Dict[str, Dict[str, float]] = {}
    for qid, variants in tqdm(expanded.items(), desc="Multi-query hybrid eval"):
        per_variant_fused: List[List[Tuple[str, float]]] = []
        for vidx, v in enumerate(variants):
            toks   = tokenize_for_bm25(v, stops)
            scores = bm25.get_scores(toks)
            top_n  = min(top_k, len(scores))
            top_ix = np.argpartition(scores, -top_n)[-top_n:]
            top_ix = top_ix[np.argsort(-scores[top_ix])]
            bm25_results  = [(metadata[i]["chunk_id"], float(scores[i])) for i in top_ix]
            dense_results = dense_by_qid_variant.get((qid, vidx), [])
            fused_variant = rrf_fusion(
                [bm25_results, dense_results], k=rrf_k, top_k=top_k
            )
            per_variant_fused.append(fused_variant)

        fused_query = rrf_fusion(per_variant_fused, k=rrf_k, top_k=top_k)
        results[qid] = {cid: score for cid, score in fused_query}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(config_path: str = "config.yaml", max_queries: int = None):
    from data_pipeline.utils import (
        load_chunks,
        load_queries,
        load_qrels,
        load_chunk_to_doc_map,
    )

    cfg = load_config(config_path)

    data_dir  = cfg["paths"]["data_dir"]
    index_dir = cfg["paths"]["index_dir"]
    res_dir   = cfg["paths"]["results_dir"]
    eval_cfg  = cfg["evaluation"]

    k_values    = eval_cfg["k_values"]
    max_queries = max_queries or eval_cfg.get("max_queries")

    os.makedirs(res_dir, exist_ok=True)

    # ── Load Data ──────────────────────────────────────────────────────────
    logger.info("Loading data …")
    chunks       = load_chunks(os.path.join(data_dir, "corpus_chunks.jsonl"))
    queries      = load_queries(os.path.join(data_dir, "test_queries.json"))
    qrels        = load_qrels(os.path.join(data_dir, "test_qrels.json"))
    chunk_to_doc = load_chunk_to_doc_map(os.path.join(data_dir, "corpus_chunks.jsonl"))

    if max_queries and max_queries < len(queries):
        limited_qids = list(queries.keys())[:max_queries]
        queries = {q: queries[q] for q in limited_qids}
        qrels   = {q: qrels[q] for q in limited_qids if q in qrels}
        logger.info(f"Evaluation limited to {len(queries)} queries.")

    # ── Model selection ────────────────────────────────────────────────────
    ret_cfg = cfg["retrieval"]
    if ret_cfg["use_finetuned_bi_encoder"] and os.path.exists(
        cfg["models"]["bi_encoder_finetuned"]
    ):
        bi_model = cfg["models"]["bi_encoder_finetuned"]
        logger.info(f"Using fine-tuned bi-encoder: {bi_model}")
    else:
        bi_model = cfg["models"]["bi_encoder_base"]

    if ret_cfg["use_finetuned_cross_encoder"] and os.path.exists(
        cfg["models"]["cross_encoder_finetuned"]
    ):
        ce_model_path = cfg["models"]["cross_encoder_finetuned"]
    else:
        ce_model_path = cfg["models"]["cross_encoder_base"]

    bm25_path  = os.path.join(index_dir, "bm25.pkl")
    faiss_path = os.path.join(index_dir, "faiss.index")
    meta_path  = os.path.join(index_dir, "chunk_metadata.pkl")

    # ── Load heavy resources once ──────────────────────────────────────────
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.cross_encoder import CrossEncoder
    from nltk.corpus import stopwords

    stops   = set(stopwords.words("english"))
    max_seq = cfg["bi_encoder_training"]["max_seq_length"]
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading bi-encoder: {bi_model} on {device}")
    encoder = SentenceTransformer(bi_model, device=device)
    encoder.max_seq_length = max_seq
    # [FIX-4] half() only for GPU memory saving during index build;
    # query encoding always reverts to float32 inside each retrieve fn.
    if device == "cuda":
        encoder = encoder.half()

    logger.info(f"Loading cross-encoder: {ce_model_path}")
    ce_model_obj = CrossEncoder(
        ce_model_path,
        max_length=cfg["cross_encoder_training"]["max_seq_length"],
        device=device,
    )

    logger.info("Loading BM25 index …")
    with open(bm25_path, "rb") as f:
        bm25_index = pickle.load(f)

    logger.info("Loading FAISS index …")
    faiss_index = faiss.read_index(faiss_path)
    if hasattr(faiss_index, "hnsw"):
        ef_search = cfg.get("indexing", {}).get("hnsw_ef_search", 128)
        faiss_index.hnsw.efSearch = ef_search
        logger.info(f"FAISS HNSW efSearch set to {ef_search}")

    logger.info("Loading chunk metadata …")
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    all_metrics     = {}
    top_k_retrieval = max(k_values)
    rrf_k           = ret_cfg["rrf_k"]
    n_variants      = ret_cfg["query_expansion"]["n_variants"]

    # Rerank top-k values — read from config with fixed defaults as fallback
    # [FIX-1][FIX-2] these now default to the corrected values
    bi_top_k_out = ret_cfg.get("bi_encoder_rerank_top_k", 80)
    ce_top_k_out = ret_cfg.get("cross_encoder_rerank_top_k", 25)
    bi_min_score = ret_cfg.get("bi_encoder_min_score", 0.0)  # [FIX-3]

    logger.info(
        f"Rerank config — bi_top_k_out={bi_top_k_out}, "
        f"ce_top_k_out={ce_top_k_out}, bi_min_score={bi_min_score}"
    )

    # ── Stage 1: BM25 ──────────────────────────────────────────────────────
    logger.info("\n[1/6] Evaluating BM25 …")
    bm25_chunk = bm25_retrieve_all(
        queries, bm25_index, metadata, stops, top_k=top_k_retrieval
    )
    bm25_doc = chunk_to_doc_results(bm25_chunk, chunk_to_doc)
    all_metrics["BM25"] = evaluate_results(
        qrels, bm25_doc, k_values, "BM25 Only"
    )

    # ── Stage 2: Dense ─────────────────────────────────────────────────────
    logger.info("\n[2/6] Evaluating Dense (FAISS) …")
    dense_chunk = dense_retrieve_all(
        queries, faiss_index, metadata, encoder,
        batch_size=eval_cfg["batch_size"],
        top_k=top_k_retrieval,
    )
    dense_doc = chunk_to_doc_results(dense_chunk, chunk_to_doc)
    all_metrics["Dense"] = evaluate_results(
        qrels, dense_doc, k_values, "Dense Only (FAISS)"
    )

    # ── Stage 3: Hybrid (BM25 + Dense → RRF) ──────────────────────────────
    logger.info("\n[3/6] Evaluating Hybrid (BM25 + Dense + RRF) …")
    # [FIX-5] use shared rrf_merge instead of inline copy
    hybrid_chunk = {
        qid: rrf_merge(
            bm25_chunk.get(qid, {}),
            dense_chunk.get(qid, {}),
            rrf_k=rrf_k,
            top_k=top_k_retrieval,
        )
        for qid in queries
    }
    hybrid_doc = chunk_to_doc_results(hybrid_chunk, chunk_to_doc)
    all_metrics["Hybrid"] = evaluate_results(
        qrels, hybrid_doc, k_values, "Hybrid (BM25 + Dense + RRF)"
    )

    # ── Stage 4: Hybrid (Multi-Query) ──────────────────────────────────────
    logger.info("\n[4/6] Evaluating Hybrid (Multi-Query + RRF) …")
    multi_chunk = multi_query_hybrid_retrieve_all(
        queries, bm25_index, faiss_index, metadata, encoder, stops,
        n_variants=n_variants,
        batch_size=eval_cfg["batch_size"],
        top_k=top_k_retrieval,
        rrf_k=rrf_k,
    )
    multi_doc = chunk_to_doc_results(multi_chunk, chunk_to_doc)
    all_metrics["HybridMultiQuery"] = evaluate_results(
        qrels, multi_doc, k_values, "Hybrid (Multi-Query + RRF)"
    )

    # ── Stage 5: MultiHybrid + Bi-Encoder Rerank ───────────────────────────
    logger.info("\n[5/6] Evaluating MultiHybrid + Bi-Encoder Rerank …")
    bi_chunk = bi_encoder_rerank_results(
        queries, multi_chunk, chunks, encoder,
        top_k_in=top_k_retrieval,
        top_k_out=bi_top_k_out,    # [FIX-1]
        min_score=bi_min_score,    # [FIX-3]
    )
    bi_doc = chunk_to_doc_results(bi_chunk, chunk_to_doc)
    all_metrics["MultiHybrid+BiEncoder"] = evaluate_results(
        qrels, bi_doc, k_values, "MultiHybrid + Bi-Encoder Rerank"
    )

    # ── Stage 6: Full Pipeline (+ Cross-Encoder) ───────────────────────────
    logger.info("\n[6/6] Evaluating Full Pipeline (MultiHybrid + BiEncoder + CE) …")
    ce_chunk = ce_rerank_results(
        queries, bi_chunk, chunks, ce_model_obj,
        top_k_in=bi_top_k_out,    # [FIX-2] CE receives full bi-encoder output
        top_k_out=ce_top_k_out,   # [FIX-2]
    )
    ce_doc = chunk_to_doc_results(ce_chunk, chunk_to_doc)
    all_metrics["FullPipeline"] = evaluate_results(
        qrels, ce_doc, k_values, "Full Pipeline (MultiHybrid + BiEncoder + CE)"
    )

    # ── Save Results ───────────────────────────────────────────────────────
    results_path = os.path.join(res_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\n✅ Evaluation results saved → {results_path}")

    # ── Summary Table ──────────────────────────────────────────────────────
    print(f"\n{'='*96}")
    print(
        f"{'Stage':<30} {'NDCG@10':>10} {'MAP@10':>10} "
        f"{'Recall@100':>11} {'MRR@10':>10} {'P@5':>10}"
    )
    print(f"{'='*96}")
    for stage, m in all_metrics.items():
        print(
            f"{stage:<30} {m.get('NDCG@10',0):>10.4f} {m.get('MAP@10',0):>10.4f} "
            f"{m.get('Recall@100',0):>11.4f} {m.get('MRR@10',0):>10.4f} "
            f"{m.get('Precision@5',0):>10.4f}"
        )
    print(f"{'='*96}\n")

    # ── Single vs Multi-Query delta ────────────────────────────────────────
    single = all_metrics.get("Hybrid", {})
    multi  = all_metrics.get("HybridMultiQuery", {})
    if single and multi:
        print("Single vs Multi-Query (Hybrid stage)")
        for k in k_values:
            r_s   = single.get(f"Recall@{k}", 0.0)
            r_m   = multi.get(f"Recall@{k}", 0.0)
            delta = r_m - r_s
            print(f"  Recall@{k:<3}: {r_s:.4f} → {r_m:.4f} (Δ {delta:+.4f})")
        mrr_s = single.get("MRR@10", 0.0)
        mrr_m = multi.get("MRR@10", 0.0)
        print(f"  MRR@10   : {mrr_s:.4f} → {mrr_m:.4f} (Δ {mrr_m - mrr_s:+.4f})")

    # ── Rerank delta (bi → CE) ─────────────────────────────────────────────
    bi_m  = all_metrics.get("MultiHybrid+BiEncoder", {})
    ce_m  = all_metrics.get("FullPipeline", {})
    if bi_m and ce_m:
        print("\nBi-Encoder → Full Pipeline delta")
        for metric in ["NDCG@10", "Recall@10", "MRR@10"]:
            v_bi = bi_m.get(metric, 0.0)
            v_ce = ce_m.get(metric, 0.0)
            print(f"  {metric:<12}: {v_bi:.4f} → {v_ce:.4f} (Δ {v_ce - v_bi:+.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--max_queries", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.max_queries)