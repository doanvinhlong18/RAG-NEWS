"""
rag_inference.py
================
End-to-end RAG inference: query → retrieved docs → grounded answer.

Pipeline:
  1. Full retrieval  (retrieval_pipeline.RetrievalPipeline)
  2. Answer synthesis (generator.AnswerGenerator via llm_factory.build_generator)
     - Context compression: dedup, rank by CE, truncate
     - Intent-aware prompt assembly
     - Groq LLM generation with retry + fallback models
     - Preamble removal + sentence trimming

Run:
  python rag_inference.py --query "Who won the 2016 US election?"
  python rag_inference.py --query "..." --config config.yaml
"""

import os
import json
import logging
import argparse
from typing import Dict

import yaml
from generator import GenerationResult
from llm_factory import build_generator

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Full RAG Inference
# ---------------------------------------------------------------------------
class RAGPipeline:
    """End-to-end RAG pipeline."""

    def __init__(self, config_path: str = "config.yaml"):
        from retrieval_pipeline import RetrievalPipeline

        self.cfg       = load_config(config_path)
        self.retriever = RetrievalPipeline(config_path)
        self.generator = build_generator(self.cfg)

    def run(self, query: str, verbose: bool = False) -> Dict:
        """
        Full pipeline run.

        Returns:
            {
              "query":      str,
              "retrieved":  List[Dict],   # raw reranked docs
              "answer":     str,
              "used_docs":  List[str],
              "citations":  List[str],
              "confidence": float,
              "fallback":   bool,
            }
        """
        import time
        t_start = time.time()
        logger.info(f"Query: {query}")

        # 1. Retrieval (intent classification happens inside)
        retrieved = self.retriever.retrieve(query, verbose=verbose)

        # 2. Get classified intent from the query processor
        intent = getattr(self.retriever.query_proc, "last_intent", "general")
        logger.info(f"Intent: {intent}")

        # 3. Answer generation (context prep + intent-aware prompt + LLM + post-process)
        t_gen = time.time()
        result: GenerationResult = self.generator.generate(
            query    = query,
            raw_docs = retrieved,
            intent   = intent,
        )
        gen_time = time.time() - t_gen

        logger.info(
            f"\n[RAG End-to-End Timing Summary]\n"
            f"  Query Expansion    : {self.retriever.last_expand_time:.3f}s\n"
            f"  BM25 Retrieval     : {self.retriever.last_bm25_time:.3f}s\n"
            f"  FAISS Retrieval    : {self.retriever.last_dense_time:.3f}s\n"
            f"  Metadata Join      : {self.retriever.last_join_time:.3f}s\n"
            f"  Bi-Encoder Rerank  : {self.retriever.last_bi_time:.3f}s\n"
            f"  Cross-Encoder Rerank: {self.retriever.last_ce_time:.3f}s\n"
            f"  LLM Generation     : {gen_time:.3f}s\n"
            f"  -------------------------\n"
            f"  End-to-End Total   : {time.time() - t_start:.3f}s"
        )

        return {
            "query":      query,
            "intent":     intent,
            "retrieved":  retrieved,
            "answer":     result.answer,
            "used_docs":  result.used_docs,
            "citations":  result.citations,
            "confidence": result.confidence,
            "fallback":   result.fallback,
        }



# ---------------------------------------------------------------------------
# Output Printer
# ---------------------------------------------------------------------------
def print_result(result: Dict):
    W = 70
    print(f"\n{'='*W}")
    print(f"QUERY      : {result['query']}")
    print(f"INTENT     : {result.get('intent', 'general')}")
    print(f"CONFIDENCE : {result['confidence']:.4f}{'  [fallback]' if result['fallback'] else ''}")
    print(f"CITATIONS  : {result['citations'] or '—'}")
    print(f"{'─'*W}")
    print("RETRIEVED CONTEXT:")
    for i, doc in enumerate(result["retrieved"], 1):
        ce     = doc.get("ce_score",  "n/a")
        rrf    = doc.get("rrf_score", 0.0)
        ce_str = f"{ce:.4f}" if isinstance(ce, float) else str(ce)
        used   = "✓" if doc.get("doc_id") in result["used_docs"] else " "
        print(f"\n  [{i}]{used} doc_id={doc.get('doc_id','?')} | CE={ce_str} | RRF={rrf:.4f}")
        print(f"      Title : {doc.get('title', '')[:80]}")
        print(f"      Text  : {doc.get('chunk_text', '')[:200]} …")
    print(f"\n{'─'*W}")
    print(f"ANSWER :\n{result['answer']}")
    print(f"{'='*W}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RAG Inference Pipeline")
    parser.add_argument("--query",   required=True, help="Input query")
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    pipeline = RAGPipeline(args.config)
    result   = pipeline.run(args.query, verbose=args.verbose)
    print_result(result)

    # Save result
    results_dir = load_config(args.config)["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, "last_inference.json")
    save_result = {
        "query":      result["query"],
        "answer":     result["answer"],
        "used_docs":  result["used_docs"],
        "citations":  result["citations"],
        "confidence": result["confidence"],
        "fallback":   result["fallback"],
        "retrieved": [
            {k: v for k, v in doc.items() if isinstance(v, (str, int, float, bool))}
            for doc in result["retrieved"]
        ],
    }
    with open(result_path, "w") as f:
        json.dump(save_result, f, indent=2, ensure_ascii=False)
    logger.info(f"Result saved → {result_path}")




if __name__ == "__main__":
    main()
