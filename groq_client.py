"""
groq_client.py
==============
Production Groq API client using the OpenAI-compatible endpoint.

Handles:
    - automatic fallback through model queue on hard failures
    - per-attempt retry with exponential backoff + ±20% jitter
    - wall-clock timeout enforced by the httpx layer via the OpenAI SDK
    - Groq-specific "try again in Xs" retry-after parsing
    - error classification: rate_limit / auth / quota / timeout / connection / other
    - detailed per-request logging: model, attempt, latency, token usage

Install:
    pip install openai>=1.30.0 httpx python-dotenv
"""
from __future__ import annotations

import logging
import os
import random
import re
import time
from typing import Iterator, List

from openai import OpenAI, APIStatusError, APIConnectionError, APITimeoutError

from config import GeneratorConfig

logger = logging.getLogger(__name__)


class GroqClient:
    """
    OpenAI-compatible client pointed at Groq's inference API.

    Model queue: primary_model → fallback_models[0] → fallback_models[1] → …
    A model is skipped only on hard failures (connection, timeout, server error).
    Auth and quota errors abort immediately — retrying a different model won't help.
    """

    def __init__(
        self,
        api_key:         str,
        base_url:        str,
        primary_model:   str,
        fallback_models: List[str],
        temperature:     float = 0.1,
        top_p:           float = 0.9,
        max_tokens:      int   = 256,
        timeout_seconds: float = 60.0,
        max_retries:     int   = 3,
    ):
        self.primary_model   = primary_model
        self.fallback_models = list(fallback_models)
        self.temperature     = temperature
        self.top_p           = top_p
        self.max_tokens      = max_tokens
        self.max_retries     = max_retries

        # max_retries=0 because we manage the retry loop ourselves.
        self._client = OpenAI(
            api_key     = api_key,
            base_url    = base_url,
            timeout     = timeout_seconds,
            max_retries = 0,
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: GeneratorConfig) -> "GroqClient":
        key = os.getenv(cfg.api_key_env)
        if not key:
            raise ValueError(
                f"API key not found. "
                f"Set {cfg.api_key_env}=your_key in .env or the environment."
            )
        return cls(
            api_key         = key,
            base_url        = cfg.base_url,
            primary_model   = cfg.primary_model,
            fallback_models = cfg.fallback_models,
            temperature     = cfg.temperature,
            top_p           = cfg.top_p,
            max_tokens      = cfg.max_tokens,
            timeout_seconds = cfg.timeout_seconds,
            max_retries     = cfg.max_retries,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def complete(self, messages: List[dict]) -> str:
        """
        Send a chat completion request.
        Tries primary_model first; on hard failure walks through fallback_models.
        Returns the generated text string.
        """
        model_queue  = [self.primary_model] + self.fallback_models
        last_exc: Exception = RuntimeError("No models attempted")

        for model in model_queue:
            try:
                return self._complete_with_retry(messages, model)
            except Exception as exc:
                category = self._classify_error(exc)
                if category in ("auth", "quota"):
                    raise   # no point trying other models
                logger.warning(
                    f"[GroqClient] model={model} failed ({category}) "
                    f"— trying next fallback | {exc}"
                )
                last_exc = exc

        raise last_exc

    def stream(self, messages: List[dict]) -> Iterator[str]:
        """Yield text chunks as they arrive (for SSE / streaming endpoints)."""
        t0 = time.time()
        try:
            response = self._client.chat.completions.create(
                model       = self.primary_model,
                messages    = messages,
                temperature = self.temperature,
                top_p       = self.top_p,
                max_tokens  = self.max_tokens,
                stream      = True,
            )
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
            logger.info(
                f"[GroqClient] stream OK  model={self.primary_model}"
                f"  elapsed={time.time()-t0:.2f}s"
            )
        except Exception as exc:
            logger.error(f"[GroqClient] stream error: {exc}")
            raise

    # ── Internal ──────────────────────────────────────────────────────────────

    def _complete_with_retry(self, messages: List[dict], model: str) -> str:
        last_exc: Exception = RuntimeError("No attempts made")

        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                response = self._client.chat.completions.create(
                    model       = model,
                    messages    = messages,
                    temperature = self.temperature,
                    top_p       = self.top_p,
                    max_tokens  = self.max_tokens,
                    stream      = False,
                )
                elapsed = time.time() - t0
                usage   = response.usage
                logger.info(
                    f"[GroqClient] OK  model={model}  attempt={attempt+1}"
                    f"  elapsed={elapsed:.2f}s"
                    f"  prompt_tokens={getattr(usage, 'prompt_tokens', '?')}"
                    f"  completion_tokens={getattr(usage, 'completion_tokens', '?')}"
                )
                text = (response.choices[0].message.content or "").strip()
                if not text:
                    raise ValueError("Empty response returned by model")
                return text

            except Exception as exc:
                elapsed  = time.time() - t0
                category = self._classify_error(exc)
                last_exc = exc

                if category in ("auth", "quota"):
                    raise   # unrecoverable — don't retry

                if attempt < self.max_retries - 1:
                    delay = 2.0 * (2 ** attempt)
                    delay *= 0.8 + 0.4 * random.random()   # ±20% jitter

                    # Groq includes "try again in Xs" in rate-limit error messages.
                    if category == "rate_limit":
                        m = re.search(
                            r"(?:try again in|retry after)\s*(\d+\.?\d*)\s*s",
                            str(exc), re.IGNORECASE,
                        )
                        if m:
                            delay = max(delay, float(m.group(1)) + 1.0)

                    logger.warning(
                        f"[GroqClient] {category}  model={model}"
                        f"  attempt={attempt+1}/{self.max_retries}"
                        f"  elapsed={elapsed:.2f}s — retry in {delay:.1f}s | {exc}"
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"[GroqClient] all {self.max_retries} attempts failed"
                        f"  model={model}: {exc}"
                    )

        raise last_exc

    # ── Error classification ──────────────────────────────────────────────────

    @staticmethod
    def _classify_error(exc: Exception) -> str:
        if isinstance(exc, APIStatusError):
            if exc.status_code == 429:  return "rate_limit"
            if exc.status_code in (401, 403):  return "auth"
            if exc.status_code == 402:  return "quota"
        if isinstance(exc, APITimeoutError):    return "timeout"
        if isinstance(exc, APIConnectionError): return "connection"

        msg = str(exc).lower()
        if any(k in msg for k in ("429", "rate limit", "rate_limit", "resource_exhausted")):
            return "rate_limit"
        if any(k in msg for k in ("401", "403", "unauthorized", "invalid api key", "api_key")):
            return "auth"
        if any(k in msg for k in ("402", "quota", "insufficient_quota")):
            return "quota"
        if any(k in msg for k in ("timeout", "timed out", "deadline")):
            return "timeout"
        return "other"
