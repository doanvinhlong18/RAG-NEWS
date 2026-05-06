"""
Tiny demo for multi_query_retriever functions (no external dependencies).
"""

from multi_query_retriever import expand_query, retrieve_multi_query


def bm25_search(_q):
    return [("doc1", 10.0), ("doc2", 9.0), ("doc3", 8.0)]


def dense_search(_q):
    return [("doc2", 0.9), ("doc4", 0.8), ("doc1", 0.7)]


def main():
    query = "stock market crash"
    variants = expand_query(query, n_variants=3)
    fused = retrieve_multi_query(variants, bm25_search=bm25_search, dense_search=dense_search, top_k=5, rrf_k=60)

    print("Query:", query)
    print("Variants:", variants)
    print("Fused:", fused)


if __name__ == "__main__":
    main()

