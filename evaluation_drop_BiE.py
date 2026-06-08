"""
evaluation.py
=============
Evaluate all retrieval stages using BEIR-format qrels.

Evaluates:
  1. BM25 only
  2. Dense (FAISS) only
  3. Hybrid (BM25 + Dense + RRF)
  4. HybridMultiQuery (BM25 + Dense + RRF + query expansion)
  5. HybridMultiQuery + Cross-Encoder Rerank  ← production pipeline

Thay đổi kiến trúc so với bản cũ:
  [ARCH] Bỏ hoàn toàn bước Bi-Encoder rerank trung gian.
         CE rerank thẳng từ HybridMultiQuery pool (top 100).
         Lý do: Dense NDCG@10 = 0.289 < BM25 NDCG@10 = 0.369 cho thấy
         bi-encoder chưa học được domain. Rerank theo embedding yếu hơn
         BM25 làm giảm chất lượng ở mọi metric. CE không bị ảnh hưởng
         vì đọc toàn text, không phụ thuộc embedding quality.
  [FIX-1] CE top_k_out nâng từ 5 → 25, tránh cap cứng tại k=5
           khiến NDCG/Recall/MAP ở k>5 bằng nhau hoàn toàn.
  [FIX-2] Query encoding luôn gọi .float() dù model lưu ở half(),
           tránh cosine similarity drift trên query vectors ngắn.
  [FIX-3] rrf_merge dùng rrf_fusion từ multi_query_retriever,
           không copy-paste inline như bản gốc.
  [FIX-4] compute_precision skip query không có relevant doc
           thay vì append 0, tránh inflate denominator.

Metrics:
  - NDCG@{1,3,5,10,100}
  - MAP@{1,3,5,10,100}
  - Recall@{1,3,5,10,100}
  - Precision@{1,3,5,10,100}
  - MRR@10

Config keys mới (thêm vào config.yaml nếu muốn override):
  retrieval.ce_rerank_top_k_in  : số chunk đưa vào CE (default 100)
  retrieval.cross_encoder_rerank_top_k : số chunk giữ lại sau CE (default 25)

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
    Map chunk-level scores → doc-level bằng cách lấy max score mỗi doc mỗi query.

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
    """BM25 retrieval cho tất cả queries. Returns {qid: {chunk_id: score}}."""
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
    Dense FAISS retrieval. Returns {qid: {chunk_id: score}}.

    [FIX-2] encoder.float() trước khi encode để tránh cosine drift
    khi model đang ở half() precision trên GPU.
    """
    qids    = list(queries.keys())
    q_texts = [queries[q] for q in qids]

    logger.info(f"Encoding {len(q_texts)} queries (float32) …")
    q_embs = encoder.float().encode(
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
    ndcgs = []
    for qid in qrels:
        if qid not in results or not results[qid]:
            ndcgs.append(0.0)
            continue
        relevant     = qrels[qid]
        ranked       = sorted(results[qid].items(), key=lambda x: -x[1])[:k]
        ideal_scores = sorted(relevant.values(), reverse=True)[:k]
        dcg  = sum(relevant.get(d, 0) / np.log2(r + 1) for r, (d, _) in enumerate(ranked, 1))
        idcg = sum(v / np.log2(r + 1) for r, v in enumerate(ideal_scores, 1))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(ndcgs))


def compute_recall(qrels: Dict, results: Dict, k: int) -> float:
    recalls = []
    for qid in qrels:
        relevant = set(d for d, s in qrels[qid].items() if s > 0)
        if not relevant:
            continue
        if qid not in results or not results[qid]:
            recalls.append(0.0)
            continue
        retrieved = set(d for d, _ in sorted(results[qid].items(), key=lambda x: -x[1])[:k])
        recalls.append(len(relevant & retrieved) / len(relevant))
    return float(np.mean(recalls)) if recalls else 0.0


def compute_map(qrels: Dict, results: Dict, k: int) -> float:
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
        for rank, (doc_id, _) in enumerate(ranked, 1):
            if doc_id in relevant:
                num_hits += 1
                ap += num_hits / rank
        aps.append(ap / min(len(relevant), k))
    return float(np.mean(aps)) if aps else 0.0


def compute_precision(qrels: Dict, results: Dict, k: int) -> float:
    """
    [FIX-4] Skip queries không có relevant doc thay vì append 0.
    """
    precs = []
    for qid in qrels:
        relevant = set(d for d, s in qrels[qid].items() if s > 0)
        if not relevant:
            continue  # [FIX-4]
        if qid not in results:
            precs.append(0.0)
            continue
        retrieved = [d for d, _ in sorted(results[qid].items(), key=lambda x: -x[1])[:k]]
        precs.append(sum(1 for d in retrieved if d in relevant) / k)
    return float(np.mean(precs)) if precs else 0.0


def compute_mrr(qrels: Dict, results: Dict, k: int = 10) -> float:
    mrrs = []
    for qid in qrels:
        if qid not in results:
            mrrs.append(0.0)
            continue
        relevant = set(d for d, s in qrels[qid].items() if s > 0)
        ranked   = sorted(results[qid].items(), key=lambda x: -x[1])[:k]
        rr = next((1.0 / r for r, (d, _) in enumerate(ranked, 1) if d in relevant), 0.0)
        mrrs.append(rr)
    return float(np.mean(mrrs))


def evaluate_results(
    qrels: Dict, results: Dict, k_values: List[int], name: str
) -> Dict:
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
# RRF merge helper
# [FIX-3] Dùng rrf_fusion từ multi_query_retriever.
# ---------------------------------------------------------------------------

def rrf_merge(
    bm25_r: Dict[str, float],
    dense_r: Dict[str, float],
    rrf_k: int,
    top_k: int,
) -> Dict[str, float]:
    bm25_list  = sorted(bm25_r.items(),  key=lambda x: -x[1])
    dense_list = sorted(dense_r.items(), key=lambda x: -x[1])
    return {cid: score for cid, score in rrf_fusion([bm25_list, dense_list], k=rrf_k, top_k=top_k)}


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
    Mỗi query → n_variants → BM25 + Dense per variant → RRF fuse.
    """
    from data_pipeline.utils import tokenize_for_bm25

    expanded = {qid: expand_query(q, n_variants=n_variants) for qid, q in queries.items()}

    flat_pairs: List[Tuple[str, int, str]] = [
        (qid, vidx, v)
        for qid, variants in expanded.items()
        for vidx, v in enumerate(variants)
    ]

    logger.info(f"Encoding {len(flat_pairs)} expanded queries (float32) …")
    q_embs = encoder.float().encode(  # [FIX-2]
        [v for _, _, v in flat_pairs],
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    logger.info("FAISS search (expanded queries) …")
    D, I = faiss_index.search(q_embs, top_k)

    dense_by: Dict[Tuple[str, int], List[Tuple[str, float]]] = {}
    for row, (qid, vidx, _) in enumerate(flat_pairs):
        dense_by[(qid, vidx)] = [
            (metadata[i]["chunk_id"], float(D[row][k]))
            for k, i in enumerate(I[row]) if i >= 0
        ]

    results: Dict[str, Dict[str, float]] = {}
    for qid, variants in tqdm(expanded.items(), desc="Multi-query hybrid eval"):
        per_variant: List[List[Tuple[str, float]]] = []
        for vidx, v in enumerate(variants):
            toks   = tokenize_for_bm25(v, stops)
            scores = bm25.get_scores(toks)
            top_n  = min(top_k, len(scores))
            top_ix = np.argpartition(scores, -top_n)[-top_n:]
            top_ix = top_ix[np.argsort(-scores[top_ix])]
            bm25_r  = [(metadata[i]["chunk_id"], float(scores[i])) for i in top_ix]
            dense_r = dense_by.get((qid, vidx), [])
            per_variant.append(rrf_fusion([bm25_r, dense_r], k=rrf_k, top_k=top_k))

        fused = rrf_fusion(per_variant, k=rrf_k, top_k=top_k)
        results[qid] = {cid: score for cid, score in fused}

    return results


# ---------------------------------------------------------------------------
# Cross-Encoder Rerank — trực tiếp từ HybridMultiQuery pool
# [ARCH] Không còn bi-encoder rerank trung gian.
# ---------------------------------------------------------------------------

def ce_rerank_from_hybrid(
    queries: Dict[str, str],
    hybrid_chunk_results: Dict[str, Dict[str, float]],
    chunks: Dict[str, str],
    ce_model,
    top_k_in: int = 100,
    top_k_out: int = 25,
) -> Dict[str, Dict[str, float]]:
    """
    Cross-encoder rerank thẳng từ hybrid retrieval pool.

    [ARCH] CE nhận trực tiếp top_k_in chunks từ HybridMultiQuery,
    bỏ qua bước bi-encoder rerank vốn làm giảm chất lượng khi
    Dense yếu hơn BM25 trên domain cụ thể.

    [FIX-1] top_k_out = 25 (trước là 5) để NDCG/Recall tại k > 5
    không bị flatten về cùng một giá trị.

    Args:
        queries              : {qid: query_text}
        hybrid_chunk_results : {qid: {chunk_id: rrf_score}}
        chunks               : {chunk_id: chunk_text}
        ce_model             : CrossEncoder instance
        top_k_in             : số chunks lấy từ hybrid pool đưa vào CE
        top_k_out            : số chunks giữ lại sau CE scoring

    Latency note:
        CE chạy top_k_in forward passes mỗi query. Với top_k_in=100
        và batch_size=32, mỗi query cần ~3 batches trên GPU.
        Nếu cần tốc độ hơn, giảm top_k_in xuống 50 và chấp nhận
        Recall@100 thấp hơn một chút.
    """
    reranked: Dict[str, Dict[str, float]] = {}

    for qid in tqdm(queries, desc="CE rerank (direct from hybrid)"):
        if qid not in hybrid_chunk_results:
            continue

        query      = queries[qid]
        top_chunks = sorted(
            hybrid_chunk_results[qid].items(), key=lambda x: -x[1]
        )[:top_k_in]

        valid = [(cid, chunks[cid]) for cid, _ in top_chunks if cid in chunks]
        if not valid:
            continue

        cids, texts = zip(*valid)
        scores      = ce_model.predict([(query, t) for t in texts], show_progress_bar=False)
        ranked      = sorted(zip(cids, scores.tolist()), key=lambda x: -x[1])[:top_k_out]

        reranked[qid] = {cid: float(score) for cid, score in ranked}

    return reranked


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

    # ── Load resources ─────────────────────────────────────────────────────
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.cross_encoder import CrossEncoder
    from nltk.corpus import stopwords

    stops   = set(stopwords.words("english"))
    max_seq = cfg["bi_encoder_training"]["max_seq_length"]
    device  = "cuda" if torch.cuda.is_available() else "cpu"

    # Bi-encoder: chỉ còn dùng cho Dense retrieval và multi-query encoding.
    # Không còn dùng để rerank.
    logger.info(f"Loading bi-encoder: {bi_model} on {device}")
    encoder = SentenceTransformer(bi_model, device=device)
    encoder.max_seq_length = max_seq
    if device == "cuda":
        encoder = encoder.half()  # half() để tiết kiệm VRAM; query encode sẽ gọi .float()

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

    logger.info("Loading chunk metadata …")
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    all_metrics     = {}
    top_k_retrieval = max(k_values)
    rrf_k           = ret_cfg["rrf_k"]
    n_variants      = ret_cfg["query_expansion"]["n_variants"]

    # CE config — đọc từ config.yaml, fallback về giá trị đã fix
    ce_top_k_in  = ret_cfg.get("ce_rerank_top_k_in", 100)   # chunks vào CE
    ce_top_k_out = ret_cfg.get("cross_encoder_rerank_top_k", 25)  # chunks ra [FIX-1]

    logger.info(
        f"Production pipeline: HybridMultiQuery (top {top_k_retrieval}) "
        f"→ CrossEncoder (in={ce_top_k_in}, out={ce_top_k_out})"
    )

    # ── Stage 1: BM25 ──────────────────────────────────────────────────────
    logger.info("\n[1/5] Evaluating BM25 …")
    bm25_chunk = bm25_retrieve_all(
        queries, bm25_index, metadata, stops, top_k=top_k_retrieval
    )
    bm25_doc = chunk_to_doc_results(bm25_chunk, chunk_to_doc)
    all_metrics["BM25"] = evaluate_results(qrels, bm25_doc, k_values, "BM25 Only")

    # ── Stage 2: Dense ─────────────────────────────────────────────────────
    logger.info("\n[2/5] Evaluating Dense (FAISS) …")
    dense_chunk = dense_retrieve_all(
        queries, faiss_index, metadata, encoder,
        batch_size=eval_cfg["batch_size"],
        top_k=top_k_retrieval,
    )
    dense_doc = chunk_to_doc_results(dense_chunk, chunk_to_doc)
    all_metrics["Dense"] = evaluate_results(qrels, dense_doc, k_values, "Dense Only (FAISS)")

    # ── Stage 3: Hybrid ────────────────────────────────────────────────────
    logger.info("\n[3/5] Evaluating Hybrid (BM25 + Dense + RRF) …")
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

    # ── Stage 4: HybridMultiQuery ──────────────────────────────────────────
    logger.info("\n[4/5] Evaluating HybridMultiQuery …")
    multi_chunk = multi_query_hybrid_retrieve_all(
        queries, bm25_index, faiss_index, metadata, encoder, stops,
        n_variants=n_variants,
        batch_size=eval_cfg["batch_size"],
        top_k=top_k_retrieval,
        rrf_k=rrf_k,
    )
    multi_doc = chunk_to_doc_results(multi_chunk, chunk_to_doc)
    all_metrics["HybridMultiQuery"] = evaluate_results(
        qrels, multi_doc, k_values, "Hybrid Multi-Query + RRF"
    )

    # ── Stage 5: HybridMultiQuery → CE (Production) ────────────────────────
    logger.info("\n[5/5] Evaluating FullPipeline (HybridMultiQuery → CE) …")
    ce_chunk = ce_rerank_from_hybrid(
        queries,
        multi_chunk,
        chunks,
        ce_model_obj,
        top_k_in=ce_top_k_in,
        top_k_out=ce_top_k_out,
    )
    ce_doc = chunk_to_doc_results(ce_chunk, chunk_to_doc)
    all_metrics["FullPipeline"] = evaluate_results(
        qrels, ce_doc, k_values, "Full Pipeline (HybridMultiQuery → CE)"
    )

    # ── Save Results ───────────────────────────────────────────────────────
    results_path = os.path.join(res_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\n✅ Evaluation results saved → {results_path}")

    # ── Summary Table ──────────────────────────────────────────────────────
    w = 102
    print(f"\n{'='*w}")
    print(
        f"{'Stage':<35} {'NDCG@10':>9} {'MAP@10':>9} "
        f"{'Recall@10':>10} {'Recall@100':>11} {'MRR@10':>9} {'P@5':>8}"
    )
    print(f"{'─'*w}")
    for stage, m in all_metrics.items():
        marker = " ◀" if stage == "FullPipeline" else ""
        print(
            f"{stage:<35} "
            f"{m.get('NDCG@10',0):>9.4f} "
            f"{m.get('MAP@10',0):>9.4f} "
            f"{m.get('Recall@10',0):>10.4f} "
            f"{m.get('Recall@100',0):>11.4f} "
            f"{m.get('MRR@10',0):>9.4f} "
            f"{m.get('Precision@5',0):>8.4f}"
            f"{marker}"
        )
    print(f"{'='*w}\n")

    # ── Pipeline delta: HybridMultiQuery → CE ─────────────────────────────
    mq = all_metrics.get("HybridMultiQuery", {})
    ce = all_metrics.get("FullPipeline", {})
    if mq and ce:
        print("HybridMultiQuery → CE Rerank delta:")
        for metric in [f"NDCG@{k}" for k in k_values] + ["MRR@10"]:
            v_mq  = mq.get(metric, 0.0)
            v_ce  = ce.get(metric, 0.0)
            delta = v_ce - v_mq
            arrow = "↑" if delta > 0.0001 else ("↓" if delta < -0.0001 else "=")
            print(f"  {metric:<12}: {v_mq:.4f} → {v_ce:.4f}  {arrow} ({delta:+.4f})")

    # ── BM25 vs Dense gap — gợi ý fine-tune ───────────────────────────────
    bm = all_metrics.get("BM25", {})
    dn = all_metrics.get("Dense", {})
    if bm and dn:
        gap = bm.get("NDCG@10", 0) - dn.get("NDCG@10", 0)
        status = "⚠ Nên fine-tune bi-encoder" if gap > 0.05 else "✓ Gap chấp nhận được"
        print(
            f"\nBM25 vs Dense NDCG@10 gap: {gap:.4f}  [{status}]"
        )
        if gap > 0.05:
            print(
                "  → Fine-tune bi-encoder với hard negatives trên domain data\n"
                "     sẽ thu hẹp gap và cho phép khôi phục bi-encoder rerank step."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--max_queries", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.max_queries)