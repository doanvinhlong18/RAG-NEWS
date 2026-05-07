"""
multi_query_retriever.py
========================
Multi-query retrieval helpers with intent-aware query expansion:
  - QueryIntentClassifier  — rule-based intent detection (news-domain optimized)
  - Intent-aware query expansion (statement / biography / event / causal / opinion)
  - Per-query hybrid retrieval (BM25 + Dense)
  - Reciprocal Rank Fusion (RRF) across methods and queries

Intent types
------------
  statement  "What did Trump say about immigration?"
  biography  "Who is Joe Biden?"
  event      "What happened at the Capitol on Jan 6?"
  causal     "Why did the stock market crash?"
  opinion    "How did critics react to the healthcare bill?"
  general    fallback for everything else

Example:
    from multi_query_retriever import QueryIntentClassifier, expand_query_with_intent

    clf = QueryIntentClassifier()
    intent = clf.classify("What did Biden say about Ukraine?")   # → "statement"
    variants = expand_query_with_intent("What did Biden say about Ukraine?", intent, n_variants=3)
"""

import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Default (legacy) templates — kept for backward compat
# ---------------------------------------------------------------------------
DEFAULT_TEMPLATES = [
    "{query}",
    "{query} causes",
    "{query} impact",
    "{query} effects",
    "reasons for {query}",
]


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip()


# ---------------------------------------------------------------------------
# Intent Classifier
# ---------------------------------------------------------------------------

class QueryIntentClassifier:
    """
    Rule-based news-domain query intent classifier.

    Priority order (first match wins):
        statement > biography > event > causal > opinion > general

    Notes:
    - Statement is checked first because it is most specific and most
      broken in the current pipeline.
    - Biography patterns use re.match (anchored at start).
    - Others use re.search (anywhere in the query).
    """

    # Statement: asks what someone said / announced / argued
    _STMT_RE = re.compile(
        r"\b(said|say|says|stated|announced|claimed|argued|declared|testified|"
        r"commented on|spoke about|stance on|position on|views? on|opinion on|"
        r"response to|reaction to|according to|quoted|"
        r"statement (?:about|on|regarding)|remarks? (?:about|on|regarding))\b",
        re.IGNORECASE,
    )

    # Biography: asks who / what someone is (anchored at query start)
    _BIO_RE = re.compile(
        r"^(who (?:is|was|are|were)\b|"
        r"tell me about\b|"
        r"what is \w[\w\s]{1,30} known for|"
        r"describe \b|"
        r"background of\b|biography of\b|profile of\b)",
        re.IGNORECASE,
    )

    # Event: asks what happened / when / outcome
    _EVENT_RE = re.compile(
        r"^(what happened\b|when did\b|where did\b|"
        r"what was the (?:outcome|result|aftermath|consequence)\b|"
        r"how did .{2,40}(?:happen|end|unfold)\b|"
        r"what occurred\b|what took place\b|"
        r"what (?:is|are|was|were) the (?:details|facts) of\b)",
        re.IGNORECASE,
    )

    # Causal: asks why / what caused
    _CAUSAL_RE = re.compile(
        r"^(why\b|what caused\b|what led to\b|what resulted in\b|"
        r"reasons? for\b|factors? behind\b|"
        r"what (?:are|were) the (?:reasons|causes|factors)\b|"
        r"how did .{2,40}(?:cause|lead to|result in)\b)",
        re.IGNORECASE,
    )

    # Opinion / controversy / reaction
    _OPINION_RE = re.compile(
        r"\b(controversy|controversial|debate|critics?\b|criticized|"
        r"opposed\b|supporters?\b|backlash|condemned|praised|disputed|"
        r"public reaction|opinion poll|survey says)\b",
        re.IGNORECASE,
    )

    def classify(self, query: str) -> str:
        """Return the intent label for *query*."""
        if self._STMT_RE.search(query):
            return "statement"
        if self._BIO_RE.match(query):
            return "biography"
        if self._EVENT_RE.match(query):
            return "event"
        if self._CAUSAL_RE.match(query):
            return "causal"
        if self._OPINION_RE.search(query):
            return "opinion"
        return "general"


# ---------------------------------------------------------------------------
# Subject / topic extraction for template filling
# ---------------------------------------------------------------------------

def _extract_subject_topic(query: str, intent: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract (subject, topic) from query text for intent-specific templates.

    Returns (subject, topic) — either may be None if extraction fails.
    Used to fill "{subject}" and "{topic}" placeholders in INTENT_TEMPLATES.
    """
    if intent == "statement":
        # "What did Trump say about immigration?"
        m = re.search(
            r"(?:what did|did)\s+(.+?)\s+"
            r"(?:say|state|announce|claim|argue|declare|comment|testify|respond)\s*"
            r"(?:about|on|regarding|to)?\s*(.+?)(?:\?|$)",
            query, re.IGNORECASE,
        )
        if m:
            return m.group(1).strip(), m.group(2).strip()
        # "Trump's stance on immigration" / "Trump's response to the bill"
        m = re.search(
            r"^(.+?)(?:'s)?\s+"
            r"(?:stance|position|view|opinion|response|reaction|statement|remarks?)\s+"
            r"(?:on|to|about|regarding)\s+(.+?)(?:\?|$)",
            query, re.IGNORECASE,
        )
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return None, None

    elif intent == "biography":
        # "Who is Joe Biden?" → subject = "Joe Biden"
        m = re.search(
            r"(?:who (?:is|was|are|were)|tell me about|describe|"
            r"background of|biography of|profile of)\s+(.+?)(?:\?|$)",
            query, re.IGNORECASE,
        )
        if m:
            return m.group(1).strip(), None
        return None, None

    return None, None


# ---------------------------------------------------------------------------
# Intent-specific expansion templates
# ---------------------------------------------------------------------------

# Placeholders: {query} always available; {subject} and {topic} only when extracted.
_INTENT_TEMPLATES: Dict[str, List[str]] = {
    "statement": [
        "{query}",
        "{subject} said stated announced {topic}",
        "{subject} statement remarks on {topic}",
        "{subject} comments position on {topic}",
    ],
    "biography": [
        "{query}",
        "{subject} background career history",
        "{subject} role position",
    ],
    "event": [
        "{query}",
        "{query} outcome result",
        "{query} what happened timeline",
    ],
    "causal": [
        "{query}",
        "{query} causes reasons why",
        "{query} factors led to resulted",
    ],
    "opinion": [
        "{query}",
        "{query} reaction response critics",
        "{query} debate controversy",
    ],
    "general": [
        "{query}",
        "{query} news details",
        "{query} recent developments",
    ],
}


def expand_query_with_intent(
    query:      str,
    intent:     str,
    n_variants: int = 3,
) -> List[str]:
    """
    Intent-aware query expansion.

    Generates up to *n_variants* query strings.  The original query is always
    first.  Remaining slots are filled with intent-specific keyword variants,
    falling back to DEFAULT_TEMPLATES if the intent templates are exhausted.

    Args:
        query:      Original user query (will be normalised internally).
        intent:     Intent label from QueryIntentClassifier.classify().
        n_variants: Total number of variants to return (including original).

    Returns:
        List[str] of length ≤ n_variants, original query first.
    """
    base = _normalize_query(query)
    subject, topic = _extract_subject_topic(query, intent)

    templates = _INTENT_TEMPLATES.get(intent, _INTENT_TEMPLATES["general"])
    variants: List[str] = []

    for tpl in templates:
        needs_subject = "{subject}" in tpl
        needs_topic   = "{topic}"   in tpl
        if needs_subject and not subject:
            continue
        if needs_topic and not topic:
            continue
        try:
            v = tpl.format(
                query   = base,
                subject = subject or "",
                topic   = topic   or "",
            ).strip()
        except KeyError:
            continue
        if v and v not in variants:
            variants.append(v)
        if len(variants) >= n_variants:
            break

    # Ensure original query is always first
    if not variants or variants[0] != base:
        variants = [base] + [v for v in variants if v != base]

    # Pad with default templates if still short
    if len(variants) < n_variants:
        for tpl in DEFAULT_TEMPLATES[1:]:   # skip "{query}" — already first
            v = tpl.format(query=base)
            if v not in variants:
                variants.append(v)
            if len(variants) >= n_variants:
                break

    return variants[:n_variants]


# ---------------------------------------------------------------------------
# Legacy expand_query (kept for backward compat)
# ---------------------------------------------------------------------------

def expand_query(
    query:      str,
    n_variants: int = 3,
    templates:  Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Expand a single query into semantically diverse variants.
    Legacy function — for new code prefer expand_query_with_intent().
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


# ---------------------------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------------------------

def _extract_id(item) -> str:
    if isinstance(item, (list, tuple)) and item:
        return item[0]
    return item


def rrf_fusion(
    rank_lists: List[Iterable],
    k:          int = 60,
    top_k:      Optional[int] = None,
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.

    Args:
        rank_lists: List of ranked iterables (each item is an id or (id, score) tuple).
        k:          RRF constant.
        top_k:      Optional max results to return.

    Returns:
        List of (doc_id, rrf_score) sorted by score descending.
    """
    rrf: Dict[str, float] = {}
    for ranked in rank_lists:
        for rank, item in enumerate(ranked, start=1):
            doc_id = _extract_id(item)
            rrf[doc_id] = rrf.get(doc_id, 0.0) + 1.0 / (k + rank)

    fused = sorted(rrf.items(), key=lambda x: -x[1])
    return fused[:top_k] if top_k else fused


# ---------------------------------------------------------------------------
# Multi-query hybrid retrieval
# ---------------------------------------------------------------------------

def retrieve_multi_query(
    queries:             List[str],
    bm25_search:         Callable[[str], List[Tuple[str, float]]],
    dense_search:        Optional[Callable[[str], List[Tuple[str, float]]]] = None,
    dense_batch_results: Optional[List[List[Tuple[str, float]]]] = None,
    top_k:               int = 100,
    rrf_k:               int = 60,
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
        dense_results = dense_batch_results[idx] if dense_batch_results is not None else dense_search(q)

        fused = rrf_fusion([bm25_results, dense_results], k=rrf_k, top_k=top_k)
        per_query_fused.append(fused)

    return rrf_fusion(per_query_fused, k=rrf_k, top_k=top_k)


if __name__ == "__main__":
    clf = QueryIntentClassifier()
    tests = [
        "What did Trump say about immigration?",
        "Who is Joe Biden?",
        "What happened at the Capitol on January 6?",
        "Why did the stock market crash in 2008?",
        "How did critics react to the healthcare bill?",
        "stock market crash 2008",
    ]
    for q in tests:
        intent = clf.classify(q)
        variants = expand_query_with_intent(q, intent, n_variants=3)
        print(f"[{intent:10s}] {q}")
        for i, v in enumerate(variants):
            print(f"            [{i}] {v}")
        print()
