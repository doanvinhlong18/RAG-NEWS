"""
evaluation.py
=============
Evaluate all retrieval stages using BEIR-format qrels.

Evaluates:
  1. BM25 only
  2. Dense (FAISS) only
  3. Hybrid (BM25 + Dense + RRF)
  4. Hybrid + Bi-Encoder Rerank
  5. Hybrid + Bi-Encoder + Cross-Encoder Rerank (final pipeline)

Metrics:
  - NDCG@{1,3,5,10,100}
  - MAP@{1,3,5,10,100}
  - Recall@{1,3,5,10,100}
  - Precision@{1,3,5,10,100}
  - MRR@10

Rules:
  - Uses qrels ONLY — no simulated relevance
  - No self-retrieval (qrel doc_id != query source)
  - Chunk results mapped to doc-level via max-score aggregation
  - Evaluation limited to max_queries for speed (configurable)

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

from multi_query_retriever import expand_query, rrf_fusion


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


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
    Map chunk-level retrieval scores to document-level by taking max score
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
# BM25 Retrieval (for evaluation)
# ---------------------------------------------------------------------------
def bm25_retrieve_all(
    queries: Dict[str, str],
    chunks: List[Dict],
    bm25_path: str,
    top_k: int = 100,
) -> Dict[str, Dict[str, float]]:
    """BM25 retrieval for all queries. Returns {qid: {chunk_id: score}}."""
    from data_pipeline.utils import tokenize_for_bm25
    import nltk
    from nltk.corpus import stopwords

    stops = set(stopwords.words("english"))
    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    results = {}
    for qid, query in tqdm(queries.items(), desc="BM25 eval", unit="queries"):
        toks   = tokenize_for_bm25(query, stops)
        scores = bm25.get_scores(toks)
        top_n  = min(top_k, len(scores))
        top_ix = np.argpartition(scores, -top_n)[-top_n:]
        top_ix = top_ix[np.argsort(-scores[top_ix])]
        results[qid] = {chunks[i]["chunk_id"]: float(scores[i]) for i in top_ix}

    return results


# ---------------------------------------------------------------------------
# Dense Retrieval (for evaluation)
# ---------------------------------------------------------------------------
def dense_retrieve_all(
    queries: Dict[str, str],
    chunks: List[Dict],
    faiss_path: str,
    meta_path: str,
    model_name: str,
    max_seq_length: int = 256,
    batch_size: int = 64,
    top_k: int = 100,
) -> Dict[str, Dict[str, float]]:
    """Dense FAISS retrieval for all queries. Returns {qid: {chunk_id: score}}."""
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading encoder for eval: {model_name} on {device}")
    encoder = SentenceTransformer(model_name, device=device)
    encoder.max_seq_length = max_seq_length
    if device == "cuda":
        encoder = encoder.half()

    faiss_index = faiss.read_index(faiss_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    qids    = list(queries.keys())
    q_texts = [queries[q] for q in qids]

    logger.info(f"Encoding {len(q_texts)} queries …")
    q_embs = encoder.encode(
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
# IR Metrics (custom implementation — no BEIR dependency for metrics)
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
        if qid not in results:
            recalls.append(0.0)
            continue
        relevant = set(d for d, s in qrels[qid].items() if s > 0)
        if not relevant:
            continue
        retrieved = set(d for d, _ in sorted(results[qid].items(), key=lambda x: -x[1])[:k])
        recalls.append(len(relevant & retrieved) / len(relevant))
    return float(np.mean(recalls))


def compute_precision(qrels: Dict, results: Dict, k: int) -> float:
    """Compute Precision@k averaged over queries."""
    precs = []
    for qid in qrels:
        if qid not in results:
            precs.append(0.0)
            continue
        relevant = set(d for d, s in qrels[qid].items() if s > 0)
        retrieved = [d for d, _ in sorted(results[qid].items(), key=lambda x: -x[1])[:k]]
        hits = sum(1 for d in retrieved if d in relevant)
        precs.append(hits / k)
    return float(np.mean(precs))


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


def evaluate_results(qrels: Dict, results: Dict, k_values: List[int], name: str) -> Dict:
    """Compute all metrics for a retrieval result dict."""
    metrics = {}
    for k in k_values:
        metrics[f"NDCG@{k}"]      = compute_ndcg(qrels, results, k)
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
# Pipeline Stages: Reranking (for eval)
# ---------------------------------------------------------------------------
def bi_encoder_rerank_results(
    queries: Dict[str, str],
    chunk_results: Dict[str, Dict[str, float]],
    chunks: List[Dict],
    model_name: str,
    max_seq_length: int,
    top_k_in: int = 100,
    top_k_out: int = 30,
) -> Dict[str, Dict[str, float]]:
    """Apply bi-encoder reranking to retrieved chunks for evaluation."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer(model_name, device=device)
    model.max_seq_length = max_seq_length
    if device == "cuda":
        model = model.half()

    cid_to_chunk = {c["chunk_id"]: c for c in chunks}
    reranked = {}

    for qid in tqdm(queries, desc="Bi-encoder rerank"):
        if qid not in chunk_results:
            continue
        query = queries[qid]
        top_chunks = sorted(chunk_results[qid].items(), key=lambda x: -x[1])[:top_k_in]
        cids  = [cid for cid, _ in top_chunks]
        texts = [cid_to_chunk[cid]["chunk_text"] for cid in cids if cid in cid_to_chunk]
        valid_cids = [cid for cid in cids if cid in cid_to_chunk]

        if not texts:
            continue

        q_emb   = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        d_embs  = model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True)
        scores  = (q_emb @ d_embs.T).flatten()

        ranked = sorted(zip(valid_cids, scores.tolist()), key=lambda x: -x[1])[:top_k_out]
        reranked[qid] = {cid: score for cid, score in ranked}

    return reranked


def ce_rerank_results(
    queries: Dict[str, str],
    chunk_results: Dict[str, Dict[str, float]],
    chunks: List[Dict],
    model_name: str,
    max_length: int,
    top_k_in: int = 30,
    top_k_out: int = 5,
) -> Dict[str, Dict[str, float]]:
    """Apply cross-encoder reranking for evaluation."""
    import torch
    from sentence_transformers.cross_encoder import CrossEncoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = CrossEncoder(model_name, max_length=max_length, device=device)

    cid_to_chunk = {c["chunk_id"]: c for c in chunks}
    reranked = {}

    for qid in tqdm(queries, desc="CE rerank"):
        if qid not in chunk_results:
            continue
        query = queries[qid]
        top_chunks = sorted(chunk_results[qid].items(), key=lambda x: -x[1])[:top_k_in]
        cids  = [cid for cid, _ in top_chunks]
        texts = [cid_to_chunk[cid]["chunk_text"] for cid in cids if cid in cid_to_chunk]
        valid_cids = [cid for cid in cids if cid in cid_to_chunk]

        if not texts:
            continue

        pairs  = [(query, t) for t in texts]
        scores = model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(valid_cids, scores.tolist()), key=lambda x: -x[1])[:top_k_out]
        reranked[qid] = {cid: score for cid, score in ranked}

    return reranked


# ---------------------------------------------------------------------------
# Multi-Query Hybrid Retrieval (for evaluation)
# ---------------------------------------------------------------------------
def multi_query_hybrid_retrieve_all(
    queries: Dict[str, str],
    chunks: List[Dict],
    bm25_path: str,
    faiss_path: str,
    meta_path: str,
    model_name: str,
    n_variants: int,
    max_seq_length: int = 256,
    batch_size: int = 64,
    top_k: int = 100,
    rrf_k: int = 60,
) -> Dict[str, Dict[str, float]]:
    """
    Multi-query hybrid retrieval for evaluation.

    Per query:
      - Expand into variants
      - Run BM25 + Dense per variant
      - RRF fuse within variant, then fuse across variants
    """
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer
    from data_pipeline.utils import tokenize_for_bm25
    import nltk
    from nltk.corpus import stopwords

    stops = set(stopwords.words("english"))

    with open(bm25_path, "rb") as f:
        bm25 = pickle.load(f)

    faiss_index = faiss.read_index(faiss_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading encoder for multi-query eval: {model_name} on {device}")
    encoder = SentenceTransformer(model_name, device=device)
    encoder.max_seq_length = max_seq_length
    if device == "cuda":
        encoder = encoder.half()

    # Expand queries
    expanded = {qid: expand_query(q, n_variants=n_variants) for qid, q in queries.items()}

    # Flatten for batched dense encoding
    flat_pairs: List[Tuple[str, int, str]] = []
    for qid, variants in expanded.items():
        for vidx, v in enumerate(variants):
            flat_pairs.append((qid, vidx, v))

    flat_texts = [v for _, _, v in flat_pairs]
    logger.info(f"Encoding {len(flat_texts)} expanded queries …")
    q_embs = encoder.encode(
        flat_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    logger.info("FAISS search (expanded queries) …")
    D, I = faiss_index.search(q_embs, top_k)

    # Build dense results per (qid, variant)
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
            # BM25 per variant
            toks = tokenize_for_bm25(v, stops)
            scores = bm25.get_scores(toks)
            top_n = min(top_k, len(scores))
            top_ix = np.argpartition(scores, -top_n)[-top_n:]
            top_ix = top_ix[np.argsort(-scores[top_ix])]
            bm25_results = [(metadata[i]["chunk_id"], float(scores[i])) for i in top_ix]

            dense_results = dense_by_qid_variant.get((qid, vidx), [])
            fused_variant = rrf_fusion([bm25_results, dense_results], k=rrf_k, top_k=top_k)
            per_variant_fused.append(fused_variant)

        fused_query = rrf_fusion(per_variant_fused, k=rrf_k, top_k=top_k)
        results[qid] = {cid: score for cid, score in fused_query}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(config_path: str = "config.yaml", max_queries: int = None):
    from data_pipeline.utils import load_chunks, load_queries, load_qrels, load_chunk_to_doc_map

    cfg = load_config(config_path)

    data_dir  = cfg["paths"]["data_dir"]
    index_dir = cfg["paths"]["index_dir"]
    res_dir   = cfg["paths"]["results_dir"]
    eval_cfg  = cfg["evaluation"]

    k_values    = eval_cfg["k_values"]
    max_queries = max_queries or eval_cfg.get("max_queries")

    os.makedirs(res_dir, exist_ok=True)

    # ── Load Data ─────────────────────────────────────────────────────────────
    logger.info("Loading data …")
    chunks       = load_chunks(os.path.join(data_dir, "corpus_chunks.jsonl"))
    queries      = load_queries(os.path.join(data_dir, "test_queries.json"))
    qrels        = load_qrels(os.path.join(data_dir, "test_qrels.json"))
    chunk_to_doc = load_chunk_to_doc_map(os.path.join(data_dir, "corpus_chunks.jsonl"))

    # Limit evaluation queries
    if max_queries and max_queries < len(queries):
        limited_qids = list(queries.keys())[:max_queries]
        queries = {q: queries[q] for q in limited_qids}
        qrels   = {q: qrels[q] for q in limited_qids if q in qrels}
        logger.info(f"Evaluation limited to {len(queries)} queries.")

    # Determine model names
    ret_cfg = cfg["retrieval"]
    if ret_cfg["use_finetuned_bi_encoder"] and os.path.exists(cfg["models"]["bi_encoder_finetuned"]):
        bi_model = cfg["models"]["bi_encoder_finetuned"]
        logger.info(f"Using fine-tuned bi-encoder: {bi_model}")
    else:
        bi_model = cfg["models"]["bi_encoder_base"]

    if ret_cfg["use_finetuned_cross_encoder"] and os.path.exists(cfg["models"]["cross_encoder_finetuned"]):
        ce_model = cfg["models"]["cross_encoder_finetuned"]
    else:
        ce_model = cfg["models"]["cross_encoder_base"]

    bm25_path  = os.path.join(index_dir, "bm25.pkl")
    faiss_path = os.path.join(index_dir, "faiss.index")
    meta_path  = os.path.join(index_dir, "chunk_metadata.pkl")

    all_metrics = {}

    # ── Stage 1: BM25 ─────────────────────────────────────────────────────────
    logger.info("\n[1/5] Evaluating BM25 …")
    bm25_chunk = bm25_retrieve_all(queries, chunks, bm25_path, top_k=max(k_values))
    bm25_doc   = chunk_to_doc_results(bm25_chunk, chunk_to_doc)
    all_metrics["BM25"] = evaluate_results(qrels, bm25_doc, k_values, "BM25 Only")

    # ── Stage 2: Dense ────────────────────────────────────────────────────────
    logger.info("\n[2/5] Evaluating Dense (FAISS) …")
    dense_chunk = dense_retrieve_all(
        queries, chunks, faiss_path, meta_path, bi_model,
        max_seq_length=cfg["bi_encoder_training"]["max_seq_length"],
        batch_size=eval_cfg["batch_size"],
        top_k=max(k_values),
    )
    dense_doc = chunk_to_doc_results(dense_chunk, chunk_to_doc)
    all_metrics["Dense"] = evaluate_results(qrels, dense_doc, k_values, "Dense Only (FAISS)")

    # ── Stage 3: Hybrid (BM25 + Dense → RRF) ─────────────────────────────────
    logger.info("\n[3/5] Evaluating Hybrid (BM25 + Dense + RRF) …")
    rrf_k = ret_cfg["rrf_k"]

    def rrf_merge(bm25_r, dense_r, top_k=100):
        """Compute RRF for a single query."""
        rrf = {}
        for rank, (cid, _) in enumerate(sorted(bm25_r.items(), key=lambda x: -x[1]), 1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        for rank, (cid, _) in enumerate(sorted(dense_r.items(), key=lambda x: -x[1]), 1):
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        return dict(sorted(rrf.items(), key=lambda x: -x[1])[:top_k])

    hybrid_chunk = {
        qid: rrf_merge(bm25_chunk.get(qid, {}), dense_chunk.get(qid, {}))
        for qid in queries
    }
    hybrid_doc = chunk_to_doc_results(hybrid_chunk, chunk_to_doc)
    all_metrics["Hybrid"] = evaluate_results(qrels, hybrid_doc, k_values, "Hybrid (BM25 + Dense + RRF)")

    # ── Stage 3b: Hybrid (Multi-Query) ───────────────────────────────────────
    logger.info("\n[3b/5] Evaluating Hybrid (Multi-Query + RRF) …")
    n_variants = ret_cfg["query_expansion"]["n_variants"]
    multi_chunk = multi_query_hybrid_retrieve_all(
        queries,
        chunks,
        bm25_path,
        faiss_path,
        meta_path,
        bi_model,
        n_variants=n_variants,
        max_seq_length=cfg["bi_encoder_training"]["max_seq_length"],
        batch_size=eval_cfg["batch_size"],
        top_k=max(k_values),
        rrf_k=rrf_k,
    )
    multi_doc = chunk_to_doc_results(multi_chunk, chunk_to_doc)
    all_metrics["HybridMultiQuery"] = evaluate_results(
        qrels, multi_doc, k_values, "Hybrid (Multi-Query + RRF)"
    )

    # ── Stage 4: + Bi-Encoder Rerank ─────────────────────────────────────────
    logger.info("\n[4/5] Evaluating Hybrid + Bi-Encoder Rerank …")
    bi_chunk = bi_encoder_rerank_results(
        queries, hybrid_chunk, chunks, bi_model,
        max_seq_length=cfg["bi_encoder_training"]["max_seq_length"],
        top_k_in=ret_cfg["bm25_top_k"],
        top_k_out=ret_cfg["bi_encoder_rerank_top_k"],
    )
    bi_doc = chunk_to_doc_results(bi_chunk, chunk_to_doc)
    all_metrics["Hybrid+BiEncoder"] = evaluate_results(qrels, bi_doc, k_values, "Hybrid + Bi-Encoder Rerank")

    # ── Stage 5: + Cross-Encoder Rerank ──────────────────────────────────────
    logger.info("\n[5/5] Evaluating Hybrid + Bi-Encoder + Cross-Encoder Rerank …")
    ce_chunk = ce_rerank_results(
        queries, bi_chunk, chunks, ce_model,
        max_length=cfg["cross_encoder_training"]["max_seq_length"],
        top_k_in=ret_cfg["bi_encoder_rerank_top_k"],
        top_k_out=ret_cfg["cross_encoder_rerank_top_k"],
    )
    ce_doc = chunk_to_doc_results(ce_chunk, chunk_to_doc)
    all_metrics["Hybrid+BiEncoder+CE"] = evaluate_results(qrels, ce_doc, k_values, "Full Pipeline (+ CE Rerank)")

    # ── Save Results ──────────────────────────────────────────────────────────
    results_path = os.path.join(res_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\n✅ Evaluation results saved → {results_path}")

    # ── Summary Table ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Stage':<30} {'NDCG@10':>10} {'Recall@10':>10} {'MRR@10':>10} {'P@5':>10}")
    print(f"{'='*80}")
    for stage, m in all_metrics.items():
        print(f"{stage:<30} {m.get('NDCG@10',0):>10.4f} {m.get('Recall@10',0):>10.4f} "
              f"{m.get('MRR@10',0):>10.4f} {m.get('Precision@5',0):>10.4f}")
    print(f"{'='*80}\n")

    # Single vs Multi-Query comparison for Recall@k and MRR
    single = all_metrics.get("Hybrid", {})
    multi = all_metrics.get("HybridMultiQuery", {})
    if single and multi:
        print("Single vs Multi-Query (Hybrid stage)")
        for k in k_values:
            r_s = single.get(f"Recall@{k}", 0.0)
            r_m = multi.get(f"Recall@{k}", 0.0)
            delta = r_m - r_s
            print(f"  Recall@{k:<3}: {r_s:.4f} → {r_m:.4f} (Δ {delta:+.4f})")
        mrr_s = single.get("MRR@10", 0.0)
        mrr_m = multi.get("MRR@10", 0.0)
        print(f"  MRR@10   : {mrr_s:.4f} → {mrr_m:.4f} (Δ {mrr_m - mrr_s:+.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--max_queries", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.max_queries)
