"""
rag_inference.py
================
End-to-end RAG inference: query → retrieved docs → grounded answer.

Pipeline:
  1. Full retrieval (via retrieval_pipeline.RetrievalPipeline)
  2. Context selection (top-3 to 5 chunks, deduplicated by doc)
  3. Grounded prompt assembly
  4. Answer generation (google/flan-t5-base by default)
     - Fallback: extractive summary from context

Memory:
  - flan-t5-base: ~990MB on disk, ~500MB VRAM in fp16
  - Combined with retrieval models: ~2.8GB peak VRAM

Run:
  python rag_inference.py --query "Who won the 2016 US election?"
  python rag_inference.py --query "..." --config config.yaml
"""

import os
import re
import json
import logging
import argparse
from typing import Dict, List, Optional

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Context Selection
# ---------------------------------------------------------------------------
def select_context(
    reranked_docs: List[Dict],
    top_k: int = 5,
    max_tokens: int = 450,
    deduplicate_by_doc: bool = True,
) -> List[Dict]:
    """
    Select top-k chunks for context, with optional deduplication.

    - Deduplication: if two chunks share the same doc_id, keep only the
      highest-CE-scored one (avoids context being dominated by one article)
    - Truncates combined context to max_tokens words

    Returns list of selected chunk dicts.
    """
    if not reranked_docs:
        return []

    if deduplicate_by_doc:
        seen_docs: set = set()
        deduped = []
        for doc in reranked_docs:
            doc_id = doc.get("doc_id", doc.get("chunk_id", ""))
            if doc_id not in seen_docs:
                seen_docs.add(doc_id)
                deduped.append(doc)
        docs = deduped[:top_k]
    else:
        docs = reranked_docs[:top_k]

    # Enforce token budget
    selected, token_count = [], 0
    for doc in docs:
        text   = doc.get("chunk_text", "")
        n_toks = len(text.split())
        if token_count + n_toks > max_tokens and selected:
            break
        selected.append(doc)
        token_count += n_toks

    return selected


# ---------------------------------------------------------------------------
# Prompt Assembly
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = (
    "You are a factual news assistant. Answer the question using ONLY the provided context. "
    "If the answer is not in the context, say 'I don't have enough information to answer this.'\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def build_prompt(query: str, context_docs: List[Dict]) -> str:
    """
    Build a grounded RAG prompt.
    Context passages are separated by double newlines (not '|').
    """
    passages = []
    for i, doc in enumerate(context_docs, 1):
        title = doc.get("title", "")
        text  = doc.get("chunk_text", "").strip()
        if title:
            passages.append(f"[{i}] {title}\n{text}")
        else:
            passages.append(f"[{i}] {text}")

    context_str = "\n\n".join(passages)
    return PROMPT_TEMPLATE.format(context=context_str, question=query)


# ---------------------------------------------------------------------------
# Answer Generation
# ---------------------------------------------------------------------------
class AnswerGenerator:
    """
    Generates answers using google/flan-t5-base (fp16, CPU fallback).
    Provides extractive fallback if model load fails.
    """

    def __init__(self, model_name: str = "google/flan-t5-base", device: str = "auto",
                 max_new_tokens: int = 256):
        self.model_name      = model_name
        self.max_new_tokens  = max_new_tokens
        self._model          = None
        self._tokenizer      = None
        self._device         = device

    def _resolve_device(self) -> str:
        import torch
        if self._device == "auto":
            if torch.cuda.is_available():
                free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                return "cuda" if free > 800 * 1024 ** 2 else "cpu"
            return "cpu"
        return self._device

    def _load(self):
        if self._model is not None:
            return
        from transformers import T5ForConditionalGeneration, T5Tokenizer
        import torch

        device = self._resolve_device()
        logger.info(f"Loading generator: {self.model_name} on {device} …")
        try:
            self._tokenizer = T5Tokenizer.from_pretrained(self.model_name)
            self._model     = T5ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            self._model.eval()
            self._device_resolved = device
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            self._model = None

    def generate(self, prompt: str) -> str:
        """Generate answer from prompt. Falls back to extractive on failure."""
        self._load()

        if self._model is None:
            logger.warning("Generator not loaded — using extractive fallback.")
            return self._extractive_fallback(prompt)

        import torch
        try:
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
            ).to(self._device_resolved)

            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )

            answer = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
            return answer.strip()

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("OOM during generation — moving to CPU.")
                import torch as _t
                _t.cuda.empty_cache()
                self._model = self._model.to("cpu").float()
                self._device_resolved = "cpu"
                return self.generate(prompt)
            raise

    @staticmethod
    def _extractive_fallback(prompt: str) -> str:
        """Return first sentence from the first context passage."""
        # Extract context block
        ctx_match = re.search(r"Context:\n(.*?)\n\nQuestion:", prompt, re.DOTALL)
        if ctx_match:
            ctx = ctx_match.group(1)
            sentences = re.split(r"(?<=[.!?])\s+", ctx)
            return sentences[0].strip() if sentences else ctx[:300]
        return "Unable to generate an answer from the provided context."


# ---------------------------------------------------------------------------
# Full RAG Inference
# ---------------------------------------------------------------------------
class RAGPipeline:
    """End-to-end RAG pipeline."""

    def __init__(self, config_path: str = "config.yaml"):
        from retrieval_pipeline import RetrievalPipeline

        self.cfg      = load_config(config_path)
        rag_cfg       = self.cfg["rag_inference"]

        self.retriever  = RetrievalPipeline(config_path)
        self.generator  = AnswerGenerator(
            model_name     = self.cfg["models"]["generator"],
            device         = rag_cfg["generator_device"],
            max_new_tokens = rag_cfg["max_answer_tokens"],
        )
        self.context_top_k     = rag_cfg["context_top_k"]
        self.max_context_tokens = rag_cfg["max_context_tokens"]

    def run(self, query: str, verbose: bool = False) -> Dict:
        """
        Full pipeline run.

        Returns:
            {
              "query":    str,
              "context":  List[Dict],
              "prompt":   str,
              "answer":   str,
            }
        """
        logger.info(f"Query: {query}")

        # 1. Retrieval
        reranked = self.retriever.retrieve(query, verbose=verbose)

        # 2. Context selection
        context = select_context(reranked, self.context_top_k, self.max_context_tokens)

        # 3. Prompt
        prompt = build_prompt(query, context)

        # 4. Generation
        answer = self.generator.generate(prompt)

        return {
            "query":   query,
            "context": context,
            "prompt":  prompt,
            "answer":  answer,
        }





# ---------------------------------------------------------------------------
# Example Output Printer
# ---------------------------------------------------------------------------
def print_example(result: Dict):
    print(f"\n{'='*70}")
    print(f"QUERY  : {result['query']}")
    print(f"{'─'*70}")
    print("RETRIEVED CONTEXT:")
    for i, doc in enumerate(result["context"], 1):
        ce  = doc.get("ce_score", "n/a")
        rrf = doc.get("rrf_score", 0.0)
        ce_str = f"{ce:.4f}" if isinstance(ce, float) else str(ce)
        print(f"\n  [{i}] doc_id={doc.get('doc_id','?')} | CE={ce_str} | RRF={rrf:.4f}")
        print(f"      Title : {doc.get('title', '')[:80]}")
        print(f"      Text  : {doc.get('chunk_text', '')[:250]} …")
    print(f"\n{'─'*70}")
    print(f"ANSWER :\n{result['answer']}")
    print(f"{'='*70}\n")


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
    print_example(result)

    # Save result
    results_dir = load_config(args.config)["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    result_path = os.path.join(results_dir, "last_inference.json")
    # Remove non-serializable items before saving
    save_result = {
        "query":  result["query"],
        "answer": result["answer"],
        "context": [
            {k: v for k, v in c.items() if isinstance(v, (str, int, float, bool))}
            for c in result["context"]
        ],
    }
    with open(result_path, "w") as f:
        json.dump(save_result, f, indent=2, ensure_ascii=False)
    logger.info(f"Result saved → {result_path}")




if __name__ == "__main__":
    main()
