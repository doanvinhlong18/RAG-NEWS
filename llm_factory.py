"""
llm_factory.py
==============
Single entry point to instantiate the AnswerGenerator from config.yaml.

Usage (in rag_inference.py):
    from llm_factory import build_generator
    self.generator = build_generator(self.cfg)
"""
from __future__ import annotations

import logging

from config import get_generator_config, get_rag_inference_config
from generator import AnswerGenerator
from groq_client import GroqClient

logger = logging.getLogger(__name__)


def build_generator(cfg: dict) -> AnswerGenerator:
    """
    Instantiate AnswerGenerator from a parsed config.yaml dict.
    Reads cfg["generator"] for LLM settings and cfg["rag_inference"] for context settings.
    """
    gen_cfg      = get_generator_config(cfg)
    rag_cfg      = get_rag_inference_config(cfg)
    ce_threshold = cfg.get("retrieval", {}).get("ce_score_threshold", -15.0)

    logger.info(
        f"[llm_factory] provider={gen_cfg.provider}"
        f"  primary={gen_cfg.primary_model}"
        f"  fallbacks={gen_cfg.fallback_models}"
        f"  max_tokens={gen_cfg.max_tokens}"
        f"  timeout={gen_cfg.timeout_seconds}s"
    )

    client = GroqClient.from_config(gen_cfg)

    return AnswerGenerator(
        client           = client,
        rag_cfg          = rag_cfg,
        min_ce_threshold = ce_threshold,
    )
