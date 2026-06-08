"""
config.py
=========
Typed configuration dataclasses for the generator layer.
Keeps config access explicit and IDE-friendly throughout the codebase.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import yaml


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GeneratorConfig:
    provider:        str        = "groq"
    primary_model:   str        = "llama-3.3-70b-versatile"
    fallback_models: List[str]  = field(default_factory=lambda: ["qwen-qwq-32b", "mixtral-8x7b-32768"])
    temperature:     float      = 0.1
    top_p:           float      = 0.9
    max_tokens:      int        = 256
    timeout_seconds: float      = 60.0
    max_retries:     int        = 3
    streaming:       bool       = False
    api_key_env:     str        = "GROQ_API_KEY"
    base_url:        str        = "https://api.groq.com/openai/v1"


@dataclass
class RAGInferenceConfig:
    context_top_k:        int   = 3
    max_context_tokens:   int   = 450
    max_answer_tokens:    int   = 256
    max_chunk_words:      int   = 80
    max_answer_sentences: int   = 4


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_yaml(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_generator_config(cfg: dict) -> GeneratorConfig:
    g = cfg.get("generator", {})
    return GeneratorConfig(
        provider        = g.get("provider",        "groq"),
        primary_model   = g.get("primary_model",   "llama-3.3-70b-versatile"),
        fallback_models = g.get("fallback_models", ["qwen-qwq-32b", "mixtral-8x7b-32768"]),
        temperature     = g.get("temperature",     0.1),
        top_p           = g.get("top_p",           0.9),
        max_tokens      = g.get("max_tokens",      256),
        timeout_seconds = g.get("timeout_seconds", 60.0),
        max_retries     = g.get("max_retries",     3),
        streaming       = g.get("streaming",       False),
        api_key_env     = g.get("api_key_env",     "GROQ_API_KEY"),
        base_url        = g.get("base_url",        "https://api.groq.com/openai/v1"),
    )


def get_rag_inference_config(cfg: dict) -> RAGInferenceConfig:
    r = cfg.get("rag_inference", {})
    return RAGInferenceConfig(
        context_top_k        = r.get("context_top_k",        3),
        max_context_tokens   = r.get("max_context_tokens",   450),
        max_answer_tokens    = r.get("max_answer_tokens",    256),
        max_chunk_words      = r.get("max_chunk_words",      80),
        max_answer_sentences = r.get("max_answer_sentences", 4),
    )
