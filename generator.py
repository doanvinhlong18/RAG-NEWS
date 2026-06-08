"""
generator.py
============
Answer synthesis layer for the RAG pipeline.

Pipeline:
    raw retrieved docs
        → ContextCompressor   (dedup by doc_id, sort by CE, truncate)
        → PromptAssembler     (clean [Context N] format, intent-aware system prompt)
        → GroqClient          (LLM call with retry + fallback)
        → _post_process()     (strip preamble, trim to N sentences)
        → GenerationResult

The GenerationResult schema is identical to what rag_inference.py and app.py expect,
so no changes are needed upstream.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Dict, List

from config import GeneratorConfig, RAGInferenceConfig
from groq_client import GroqClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output schema  (identical interface to the old answer_generator.GenerationResult)
# ---------------------------------------------------------------------------

@dataclass
class GenerationResult:
    answer:        str
    used_docs:     List[str]
    citations:     List[str]
    confidence:    float
    context_count: int
    fallback:      bool = False

    def to_dict(self) -> Dict:
        return {
            "answer":        self.answer,
            "used_docs":     self.used_docs,
            "citations":     self.citations,
            "confidence":    round(self.confidence, 4),
            "context_count": self.context_count,
            "fallback":      self.fallback,
        }


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a concise retrieval-augmented AI assistant. "
    "Your task is to synthesize information from the retrieved context and answer the user's question. "
    "Rules: "
    "1. Use ONLY information present in the provided context. "
    "2. Synthesize across chunks — merge relevant facts, remove duplication. "
    "3. Answer directly. Do NOT open with 'Based on the context', "
    "'According to the documents', or any similar preamble. "
    "4. Keep the answer short (2-4 sentences), factual, and information-dense. "
    "5. Do NOT mention chunks, sources, or the retrieval process. "
    "6. Do NOT hallucinate or add external knowledge. "
    "7. Always attempt to answer using whatever relevant information exists in the context, "
    "even if partial. Only respond \"I do not have enough reliable information to answer that.\" "
    "when the context contains ZERO relevant facts — not merely an imperfect or indirect answer."
)

_INTENT_ADDENDUM: Dict[str, str] = {
    "statement": (
        " The question asks what someone SAID, STATED, or ANNOUNCED. "
        "Find direct quotes or stated positions in the context. If none exist, say so explicitly."
    ),
    "biography": (
        " The question asks about a PERSON. "
        "Report their role, title, position, or background as stated in the context."
    ),
    "event": (
        " The question asks about a SPECIFIC EVENT. "
        "Report what happened, who was involved, and the outcome."
    ),
    "causal": (
        " The question asks about CAUSES or REASONS. "
        "List only factors explicitly stated in the context."
    ),
    "opinion": (
        " The question asks about REACTIONS or OPINIONS. "
        "Report only documented positions described in the context."
    ),
}


# ---------------------------------------------------------------------------
# Context Compressor
# ---------------------------------------------------------------------------

class ContextCompressor:
    """
    Cleans and ranks retrieved docs before they reach the LLM.

    Steps:
        1. Drop docs with empty/whitespace-only text
        2. Deduplicate by doc_id — keep the chunk with the highest CE score
        3. Sort by CE score descending
        4. Truncate each chunk to max_words
        5. Keep at most top_k chunks
    """

    def __init__(self, top_k: int = 3, max_words: int = 80):
        self.top_k     = top_k
        self.max_words = max_words

    def compress(self, raw_docs: List[Dict]) -> List[Dict]:
        # drop empty
        docs = [d for d in raw_docs if str(d.get("chunk_text", "")).strip()]

        # dedup by doc_id — keep highest CE score per doc
        best: Dict[str, Dict] = {}
        for d in docs:
            did = str(d.get("doc_id", ""))
            ce  = float(d.get("ce_score", 0.0) or 0.0)
            if did not in best or ce > float(best[did].get("ce_score", 0.0) or 0.0):
                best[did] = d

        # sort and slice
        docs = sorted(best.values(), key=lambda d: float(d.get("ce_score", 0.0) or 0.0), reverse=True)
        docs = docs[: self.top_k]

        # truncate
        result = []
        for d in docs:
            words = str(d.get("chunk_text", "")).split()
            text  = " ".join(words[: self.max_words])
            if len(words) > self.max_words:
                text += " …"
            result.append({**d, "chunk_text": text})

        return result


# ---------------------------------------------------------------------------
# Prompt Assembler
# ---------------------------------------------------------------------------

class PromptAssembler:
    """
    Builds clean OpenAI-style messages.

    Context format:
        [Context 1]
        <text>

        [Context 2]
        <text>

        User Question:
        <query>
    """

    @staticmethod
    def build(query: str, docs: List[Dict], intent: str = "general") -> List[Dict]:
        addendum = _INTENT_ADDENDUM.get(intent, "")
        system   = _SYSTEM_PROMPT + (" IMPORTANT:" + addendum if addendum else "")

        parts = [f"[Context {i}]\n{d.get('chunk_text', '').strip()}" for i, d in enumerate(docs, 1)]
        context_block = "\n\n".join(parts)
        user = f"{context_block}\n\nUser Question:\n{query.strip()}"

        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

_PREAMBLE_RE = re.compile(
    r"^(?:"
    r"based on (?:the )?(?:context|provided|above|documents?|passages?)[,:\s]*|"
    r"according to (?:the )?(?:context|documents?|passages?|provided information)[,:\s]*|"
    r"(?:the )?(?:context|documents?|passages?) (?:states?|mentions?|indicates?|shows?|suggests?)[,:\s]*|"
    r"from (?:the )?(?:context|provided (?:context|documents?))[,:\s]*|"
    r"as (?:mentioned|stated|noted|described) in (?:the )?(?:context|document)[,:\s]*|"
    r"the provided (?:context|documents?) (?:states?|mentions?|indicates?)[,:\s]*"
    r")",
    re.IGNORECASE,
)

_DONT_KNOW_RE = re.compile(
    r"(i don.t know|not present|cannot find|no information|not found in|"
    r"do not have enough|insufficient information)",
    re.IGNORECASE,
)


def _post_process(text: str, max_sentences: int = 4) -> str:
    text = re.sub(r"^Answer:\s*", "", text, flags=re.IGNORECASE).strip()
    text = _PREAMBLE_RE.sub("", text).strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if max_sentences > 0:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        text = " ".join(sentences[:max_sentences])
    return text.strip()


def _confidence(answer: str, docs: List[Dict]) -> float:
    if not docs:
        return 0.0
    avg_ce  = sum(float(d.get("ce_score", 0.0) or 0.0) for d in docs) / len(docs)
    ce_norm = 1.0 / (1.0 + 2.718 ** -avg_ce)

    answer_words = set(re.findall(r"\b\w+\b", answer.lower()))
    ctx_words    = set(
        w for d in docs
        for w in re.findall(r"\b\w+\b", str(d.get("chunk_text", "")).lower())
    )
    stopwords  = {"the","a","an","is","are","was","were","in","of","to","and","or","it","its"}
    content    = answer_words - stopwords
    coverage   = len(content & ctx_words) / len(content) if content else 0.0

    penalty = 0.6 if _DONT_KNOW_RE.search(answer) else 0.0
    return max(0.0, min(1.0, 0.5 * ce_norm + 0.5 * coverage - penalty))


def _extractive_fallback(docs: List[Dict], n_sentences: int = 2) -> str:
    if not docs:
        return "I do not have enough reliable information to answer that."
    text      = str(docs[0].get("chunk_text", ""))
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(sentences[:n_sentences]).strip()


# ---------------------------------------------------------------------------
# AnswerGenerator  (main orchestrator)
# ---------------------------------------------------------------------------

class AnswerGenerator:
    """
    Orchestrates context compression → prompt assembly → LLM → post-processing.

    Drop-in replacement for the old RAGAnswerGenerator.
    Interface:
        result = generator.generate(query, raw_docs, intent)
        result.answer / result.used_docs / result.confidence / result.fallback
    """

    def __init__(
        self,
        client:           GroqClient,
        rag_cfg:          RAGInferenceConfig,
        min_ce_threshold: float = -15.0,
    ):
        self.client           = client
        self.min_ce_threshold = min_ce_threshold
        self.max_sentences    = rag_cfg.max_answer_sentences
        self.compressor       = ContextCompressor(
            top_k     = rag_cfg.context_top_k,
            max_words = rag_cfg.max_chunk_words,
        )

    def generate(
        self,
        query:    str,
        raw_docs: List[Dict],
        intent:   str = "general",
    ) -> GenerationResult:
        t_start = time.time()

        # 1. Compress and filter context
        docs = self.compressor.compress(raw_docs)
        docs = [d for d in docs if float(d.get("ce_score", 0.0) or 0.0) >= self.min_ce_threshold]
        logger.info(
            f"[Generator] context_prep={time.time()-t_start:.3f}s"
            f"  chunks={len(docs)}  intent={intent}"
        )

        if not docs:
            logger.warning("[Generator] No valid context — cannot-answer fallback.")
            return GenerationResult(
                answer        = "I do not have enough reliable information to answer that.",
                used_docs     = [],
                citations     = [],
                confidence    = 0.0,
                context_count = 0,
                fallback      = True,
            )

        used_docs = [str(d.get("doc_id", "")) for d in docs]

        # 2. Build prompt
        messages = PromptAssembler.build(query, docs, intent=intent)

        # 3. LLM generation
        t_gen = time.time()
        try:
            raw_answer = self.client.complete(messages)
            logger.info(f"[Generator] llm_time={time.time()-t_gen:.3f}s")
        except Exception as exc:
            logger.error(
                f"[Generator] LLM failed ({time.time()-t_gen:.2f}s) "
                f"— extractive fallback: {exc}"
            )
            fallback_text = _extractive_fallback(docs)
            return GenerationResult(
                answer        = fallback_text,
                used_docs     = used_docs,
                citations     = [],
                confidence    = _confidence(fallback_text, docs),
                context_count = len(docs),
                fallback      = True,
            )

        # 4. Post-process
        answer     = _post_process(raw_answer, max_sentences=self.max_sentences)
        confidence = _confidence(answer, docs)

        logger.info(
            f"[Generator] total={time.time()-t_start:.3f}s"
            f"  confidence={confidence:.3f}  fallback=False"
        )
        return GenerationResult(
            answer        = answer,
            used_docs     = used_docs,
            citations     = [],
            confidence    = confidence,
            context_count = len(docs),
            fallback      = False,
        )
