"""
app.py
======
Flask web server for the RAG News demo.

Run:
    python app.py
    python app.py --config config.yaml --port 5000

The RAG pipeline loads in a background thread so the server starts immediately.
Frontend polls /health until the pipeline is ready before enabling the input.
"""

import argparse
import logging
import os
import threading
import time

from dotenv import load_dotenv
load_dotenv()  # đọc .env từ project root

# Suppress Windows symlinks warning and skip HF Hub network checks for local models
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from flask import Flask, jsonify, render_template, request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Pipeline state (shared across threads)
# ---------------------------------------------------------------------------
_pipeline       = None
_pipeline_ready = False
_pipeline_error = None
_pipeline_start = None   # epoch time when loading began


def _init_pipeline(config_path: str):
    global _pipeline, _pipeline_ready, _pipeline_error, _pipeline_start
    _pipeline_start = time.time()
    try:
        from rag_inference import RAGPipeline
        logger.info("Initialising RAG pipeline …")
        _pipeline       = RAGPipeline(config_path)
        _pipeline_ready = True
        elapsed = time.time() - _pipeline_start
        logger.info(f"Pipeline ready in {elapsed:.1f}s")
    except Exception as exc:
        _pipeline_error = str(exc)
        logger.error(f"Pipeline init failed: {exc}", exc_info=True)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    elapsed = round(time.time() - _pipeline_start, 1) if _pipeline_start else 0
    return jsonify({
        "ready":   _pipeline_ready,
        "error":   _pipeline_error,
        "elapsed": elapsed,
    })


@app.route("/ask", methods=["POST"])
def ask():
    if not _pipeline_ready:
        return jsonify({"error": "Pipeline is still loading. Please wait."}), 503

    data  = request.get_json(force=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400
    if len(query) > 1000:
        return jsonify({"error": "Query too long (max 1000 chars)."}), 400

    t0 = time.time()
    try:
        raw = _pipeline.run(query)
    except Exception as exc:
        logger.error(f"Pipeline error: {exc}", exc_info=True)
        return jsonify({"error": str(exc)}), 500

    latency = round(time.time() - t0, 2)

    # Serialise safely — numpy scalars are not JSON-serialisable
    def _safe(v):
        if isinstance(v, (str, bool, type(None))):
            return v
        try:
            return float(v)
        except (TypeError, ValueError):
            return str(v)

    result = {
        "query":      raw["query"],
        "intent":     raw.get("intent", "general"),
        "answer":     raw["answer"],
        "used_docs":  raw["used_docs"],
        "citations":  raw["citations"],
        "confidence": round(float(raw["confidence"]), 4),
        "fallback":   bool(raw["fallback"]),
        "latency_s":  latency,
        "retrieved": [
            {k: _safe(v) for k, v in doc.items()
             if isinstance(v, (str, int, float, bool, type(None)))}
            for doc in raw["retrieved"]
        ],
    }
    return jsonify(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG News Web Demo")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--port",   type=int, default=5000)
    parser.add_argument("--host",   default="0.0.0.0")
    args = parser.parse_args()

    # Load pipeline in background so Flask starts immediately
    threading.Thread(
        target=_init_pipeline,
        args=(args.config,),
        daemon=True,
    ).start()

    logger.info(f"Starting server at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
