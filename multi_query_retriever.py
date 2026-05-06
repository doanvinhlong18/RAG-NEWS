"""
multi_query_retriever.py
========================
Lightweight multi-query retrieval helpers:
  - Rule-based query expansion (low resource)
  - Per-query hybrid retrieval (BM25 + Dense)
  - Reciprocal Rank Fusion (RRF) across methods and queries

Example:
    from multi_query_retriever import expand_query, retrieve_multi_query

    variants = expand_query("stock market crash", n_variants=3)
    fused = retrieve_multi_query(
        variants,
        bm25_search=bm25_search_fn,
        dense_search=dense_search_fn,
        top_k=100,
        rrf_k=60,
    )
"""

import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_TEMPLATES = [
    "{query}",
    "{query} causes",
    "{query} impact",
    "{query} effects",
    "reasons for {query}",
]


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


def expand_query(query: str, n_variants: int = 3, templates: Optional[Sequence[str]] = None) -> List[str]:
    """
    Expand a single query into 2-3 semantically diverse variants.

    Args:
        query: Original user query.
        n_variants: Total variants to return (including the original).
        templates: Optional list of templates with "{query}" placeholder.

    Returns:
        List of unique expanded queries (original first).
    """
    if n_variants <= 1:
        return [_normalize_query(query)]

    base = _normalize_query(query)
    tpls = templates or DEFAULT_TEMPLATES

    variants: List[str] = []
    for tpl in tpls:
        v = tpl.format(query=base)
        if v not in variants:
            variants.append(v)
        if len(variants) >= n_variants:
            break

    if base not in variants:
        variants = [base] + variants[: max(0, n_variants - 1)]

    return variants[:n_variants]


def _extract_id(item) -> str:
    if isinstance(item, (list, tuple)) and item:
        return item[0]
    return item


def rrf_fusion(rank_lists: List[Iterable], k: int = 60, top_k: Optional[int] = None) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.

    Args:
        rank_lists: List of ranked lists (each list of ids or (id, score) tuples).
        k: RRF constant.
        top_k: Optional max results to return.

    Returns:
        List of (doc_id, rrf_score) sorted by score desc.
    """
    rrf: Dict[str, float] = {}
    for ranked in rank_lists:
        for rank, item in enumerate(ranked, start=1):
            doc_id = _extract_id(item)
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(rrf.items(), key=lambda x: -x[1])
    return fused[:top_k] if top_k else fused


def retrieve_multi_query(
    queries: List[str],
    bm25_search: Callable[[str], List[Tuple[str, float]]],
    dense_search: Optional[Callable[[str], List[Tuple[str, float]]]] = None,
    dense_batch_results: Optional[List[List[Tuple[str, float]]]] = None,
    top_k: int = 100,
    rrf_k: int = 60,
) -> List[Tuple[str, float]]:
    """
    Multi-query hybrid retrieval with RRF.

    Each query is retrieved independently (BM25 + Dense), fused by RRF,
    then all query-level results are fused again by RRF.
    """
    if dense_batch_results is None and dense_search is None:
        raise ValueError("Either dense_search or dense_batch_results must be provided.")

    if dense_batch_results is not None and len(dense_batch_results) != len(queries):
        raise ValueError("dense_batch_results must align with queries length.")

    per_query_fused: List[List[Tuple[str, float]]] = []

    for idx, q in enumerate(queries):
        bm25_results = bm25_search(q)
        if dense_batch_results is not None:
            dense_results = dense_batch_results[idx]
        else:
            dense_results = dense_search(q)

        fused = rrf_fusion([bm25_results, dense_results], k=rrf_k, top_k=top_k)
        per_query_fused.append(fused)

    final_fused = rrf_fusion(per_query_fused, k=rrf_k, top_k=top_k)
    return final_fused


if __name__ == "__main__":
    # Tiny self-check with toy rankings.
    q = "stock market crash"
    variants = expand_query(q, n_variants=3)
    bm25 = lambda _q: [("d1", 10.0), ("d2", 9.0), ("d3", 8.0)]
    dense = lambda _q: [("d2", 0.9), ("d4", 0.8), ("d1", 0.7)]
    fused = retrieve_multi_query(variants, bm25_search=bm25, dense_search=dense, top_k=5, rrf_k=60)
    print("variants=", variants)
    print("fused=", fused)

