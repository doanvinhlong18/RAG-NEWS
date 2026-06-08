# RAG-NEWS

Production-ready Retrieval-Augmented Generation pipeline for news question answering, built on BEIR trec-news. Optimized for low-VRAM consumer GPUs (RTX 3050 Ti, 4GB).

---

## Overview

RAG-NEWS answers questions grounded exclusively in a news corpus. It retrieves relevant passages through a multi-stage hybrid pipeline, then synthesizes a concise factual answer via a cloud LLM — no hallucination, no external knowledge.

```
Question
  → Query Expansion (3 variants, intent-aware)
  → Hybrid Retrieval (BM25 + FAISS, top-100 each)
  → RRF Fusion (top-100 merged)
  → Bi-Encoder Rerank (100 → 30)
  → Cross-Encoder Rerank (30 → 5)
  → Context Compression (5 → top-3, truncated)
  → Groq LLM Synthesis (llama-3.3-70b-versatile)
  → Grounded Answer
```

---

## Key Features

- **Hybrid retrieval** — BM25 (keyword) + FAISS (semantic) fused with Reciprocal Rank Fusion
- **Three-stage reranking** — RRF → Bi-encoder → Cross-encoder for precision at each level
- **Intent-aware query expansion** — classifies query as event / statement / biography / causal / opinion and expands with domain-specific templates
- **Groq LLM generation** — llama-3.3-70b-versatile with automatic fallback to qwen-qwq-32b and mixtral-8x7b-32768
- **Production error handling** — retry, exponential backoff, rate-limit parsing, fallback model queue
- **Flask web UI** — browser demo with real-time pipeline status polling
- **Memory-optimized** — entire pipeline fits in 4GB VRAM with fp16 + gradient accumulation

---

## Hardware Requirements

| Component | Minimum | Tested On |
|---|---|---|
| GPU VRAM | 4 GB | RTX 3050 Ti 4GB |
| RAM | 16 GB | 16 GB DDR4 |
| Storage | 20 GB | SSD recommended |
| Python | 3.10+ | 3.10 |

CPU-only mode works but FAISS search and reranking will be significantly slower.

---

## Models

| Role | Model | Finetuned | VRAM |
|---|---|---|---|
| Dense retrieval (FAISS) | `all-MiniLM-L6-v2` → `models/bi_encoder` | Yes | ~400 MB |
| Bi-encoder rerank | `models/bi_encoder` | Yes | shared |
| Cross-encoder rerank | `ms-marco-MiniLM-L-6-v2` → `models/cross_encoder` | Yes | ~200 MB |
| Answer generation | `llama-3.3-70b-versatile` (Groq API) | No | cloud |

---

## Corpus

| Property | Value |
|---|---|
| Dataset | BEIR trec-news-generated-queries |
| Source | HuggingFace `BeIR/trec-news-generated-queries` |
| Sampling | 50% stratified by doc length, seed=42 |
| Chunks | ~1.56M chunks after chunking |
| Chunk size | 256 tokens, 32-token overlap |

---

## Project Structure

```
RAG-NEWS/
│
├── app.py                        # Flask web server (GET /, GET /health, POST /ask)
├── rag_inference.py              # CLI entry point + RAGPipeline orchestrator
├── retrieval_pipeline.py         # Multi-stage retrieval: expansion → hybrid → rerank
├── multi_query_retriever.py      # Intent classifier, query expansion, RRF fusion
├── generator.py                  # Context compression, prompt assembly, answer synthesis
├── groq_client.py                # Groq API client: retry, fallback, streaming
├── llm_factory.py                # Factory: config → GroqClient + AnswerGenerator
├── config.py                     # Typed dataclasses for generator and RAG config
│
├── bi_encoder_training.py        # Fine-tune SBERT (MultipleNegativesRankingLoss)
├── cross_encoder_training.py     # Fine-tune CrossEncoder (BM25 hard negatives)
├── evaluation.py                 # Evaluate retrieval stages (NDCG, MAP, Recall, MRR)
├── run_pipeline.py               # Offline pipeline: download → chunk → index
│
├── data_pipeline/
│   ├── data_loader.py            # Download BEIR dataset, extract corpus/queries/qrels
│   ├── sampler.py                # Stratified corpus sampling
│   ├── splitter.py               # Train/val/test query split
│   ├── chunker.py                # Sliding-window document chunking
│   ├── index_builder.py          # Build BM25 (.pkl) and FAISS (.index) indexes
│   ├── sparse_bm25.py            # Memory-efficient BM25 via scipy sparse matrices
│   ├── training_dataset_builder.py  # Build training triplets for both encoders
│   └── utils.py                  # Shared text utilities
│
├── templates/
│   └── index.html                # Web UI (Flask Jinja2)
│
├── config.yaml                   # All hyperparameters and paths
├── requirements.txt              # Python dependencies
├── .env                          # API keys (not committed)
│
└── docs/
    ├── pipeline.md               # RAG inference pipeline — stage-by-stage breakdown
    ├── training.md               # Bi-encoder and cross-encoder training flow
    ├── api.md                    # Flask API reference
    ├── codebase.md               # Role of every file in the project
    ├── DATA_PIPELINE_DESIGN.md   # Offline data pipeline design
    ├── END_TO_END_GUIDE.md       # Full setup guide
    ├── SPARSE_BM25_VS_RANK_BM25.md  # BM25 implementation comparison
    └── sampling_methodology.md   # Stratified sampling methodology
```

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd RAG-NEWS
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

For CPU-only:
```bash
pip install torch torchvision torchaudio
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data (run once)

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

### 5. Set Groq API key

Get a free API key at [console.groq.com](https://console.groq.com), then edit `.env`:

```
GROQ_API_KEY=gsk_your_key_here
```

The key is loaded automatically via `python-dotenv`. Never commit `.env` — it is already in `.gitignore`.

---

## Full Pipeline (First Run)

Run these steps once to build the indexes and fine-tune the models before serving.

### Step 1 — Build data and indexes

Downloads the BEIR dataset, chunks the corpus, and builds BM25 + FAISS indexes.

```bash
python run_pipeline.py
```

Outputs written to `outputs/pipeline/`:

| File | Description |
|---|---|
| `corpus_chunks.jsonl` | ~1.56M chunks with doc_id, chunk_text, n_tokens |
| `bm25.pkl` | SparseBM25 index (scipy sparse) |
| `faiss.index` | FAISS IndexFlatIP, 1.56M × 384-dim float32 |
| `chunk_metadata.pkl` | int_pos → {chunk_id, doc_id} mapping |
| `corpus_chunks_lookup.pkl` | chunk_id → {chunk_text, title} cache |
| `train_bi_encoder.jsonl` | Training triplets for bi-encoder |
| `train_cross_encoder.jsonl` | Labeled pairs for cross-encoder |

Estimated time: **45–90 minutes** depending on hardware.

### Step 2 — Fine-tune Bi-Encoder

```bash
python bi_encoder_training.py
```

Fine-tunes `all-MiniLM-L6-v2` with `MultipleNegativesRankingLoss` on BEIR trec-news triplets.

- Effective batch: 32 (batch=8 × grad_accum=4)
- fp16, 3 epochs, max 100k samples
- Peak VRAM: ~2.5 GB
- Output: `models/bi_encoder/`

### Step 3 — Fine-tune Cross-Encoder (optional)

```bash
python cross_encoder_training.py
```

Fine-tunes `ms-marco-MiniLM-L-6-v2` with BM25 hard negatives. Disabled by default (`enabled: false` in config) since the pretrained model already performs well on news.

- Effective batch: 16 (batch=4 × grad_accum=4)
- fp16, 3 epochs, max 100k samples
- Peak VRAM: ~2.0 GB
- Output: `models/cross_encoder/`

### Step 4 — Evaluate retrieval quality (optional)

```bash
python evaluation.py --config config.yaml --max_queries 500
```

Evaluates each retrieval stage independently:

| Stage | Metrics |
|---|---|
| BM25 only | NDCG, MAP, Recall, Precision @ {1,3,5,10,100} |
| Dense (FAISS) only | + MRR@10 |
| Hybrid (BM25 + Dense + RRF) | |
| Hybrid + Bi-Encoder Rerank | |
| Full pipeline (+ Cross-Encoder) | |

---

## Running

### Web UI

```bash
python app.py
# open http://localhost:5000
```

The pipeline loads in a background thread (~15–45s). The UI polls `/health` and enables the input box when ready.

### CLI

```bash
python rag_inference.py --query "Who won the 2016 US election?"
python rag_inference.py --query "What caused the 2008 financial crisis?" --verbose
```

Results are saved to `results/last_inference.json`.

### Retrieval only (no LLM)

```bash
python retrieval_pipeline.py --query "stock market crash 2008" --verbose
```

---

## API

### `POST /ask`

```bash
curl -X POST http://localhost:5000/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "Who won the 2016 US election?"}'
```

**Response:**

```json
{
  "query":      "Who won the 2016 US election?",
  "intent":     "event",
  "answer":     "Donald Trump won the 2016 US presidential election...",
  "used_docs":  ["a80440a0", "b12345ff"],
  "citations":  [],
  "confidence": 0.7832,
  "fallback":   false,
  "latency_s":  1.42,
  "retrieved":  [...]
}
```

Full API reference: [docs/api.md](docs/api.md)

---

## Configuration

All settings live in `config.yaml`. Key sections:

```yaml
generator:
  provider: "groq"
  primary_model: "llama-3.3-70b-versatile"
  fallback_models: ["qwen-qwq-32b", "mixtral-8x7b-32768"]
  temperature: 0.1
  max_tokens: 256
  timeout_seconds: 60
  max_retries: 3

retrieval:
  bm25_top_k: 100
  dense_top_k: 100
  bi_encoder_rerank_top_k: 30
  cross_encoder_rerank_top_k: 5
  use_finetuned_bi_encoder: true
  use_finetuned_cross_encoder: true
  query_expansion:
    enabled: true
    n_variants: 3

rag_inference:
  context_top_k: 5
  max_chunk_words: 150
  max_answer_sentences: 4
```

After training, make sure to enable the finetuned models:

```yaml
retrieval:
  use_finetuned_bi_encoder: true
  use_finetuned_cross_encoder: true
```

---

## Performance

| Stage | Latency | Notes |
|---|---|---|
| FAISS index load | ~5s | One-time cold start |
| BM25 load | ~8s | One-time cold start |
| Chunk lookup (first run) | ~29s | Builds pickle cache |
| Chunk lookup (subsequent) | ~4s | Reads from cache |
| FAISS search (3 query variants) | ~4s | Linear scan 1.56M vectors |
| Bi-encoder rerank | ~0.5s | GPU, 100 → 30 |
| Cross-encoder rerank | ~0.3s | GPU, 30 → 5 |
| Groq API call | ~0.5–2s | Network, llama-3.3-70b |
| **Total (after warm-up)** | **~5–7s per query** | |

---

## Documentation

| Document | Content |
|---|---|
| [docs/pipeline.md](docs/pipeline.md) | RAG inference pipeline — full stage-by-stage breakdown |
| [docs/training.md](docs/training.md) | Bi-encoder and cross-encoder training flow |
| [docs/api.md](docs/api.md) | Flask API reference with request/response schema |
| [docs/codebase.md](docs/codebase.md) | Role of every file in the project |
| [docs/END_TO_END_GUIDE.md](docs/END_TO_END_GUIDE.md) | Detailed setup guide |
| [docs/sampling_methodology.md](docs/sampling_methodology.md) | Stratified sampling methodology |
| [docs/SPARSE_BM25_VS_RANK_BM25.md](docs/SPARSE_BM25_VS_RANK_BM25.md) | BM25 memory optimization rationale |
