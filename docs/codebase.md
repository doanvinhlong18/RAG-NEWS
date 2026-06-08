# Codebase — Vai trò từng file

## Cấu trúc project

```
RAG-NEWS/
│
├── [Inference — Online]
│   ├── app.py
│   ├── rag_inference.py
│   ├── retrieval_pipeline.py
│   ├── multi_query_retriever.py
│   ├── generator.py
│   ├── groq_client.py
│   └── llm_factory.py
│
├── [Training — Offline]
│   ├── bi_encoder_training.py
│   └── cross_encoder_training.py
│
├── [Data Pipeline — Offline, chạy một lần]
│   ├── run_pipeline.py
│   └── data_pipeline/
│       ├── data_loader.py
│       ├── sampler.py
│       ├── splitter.py
│       ├── chunker.py
│       ├── index_builder.py
│       ├── sparse_bm25.py
│       ├── training_dataset_builder.py
│       └── utils.py
│
├── [Evaluation]
│   └── evaluation.py
│
├── [Config & Infra]
│   ├── config.py
│   ├── config.yaml
│   ├── requirements.txt
│   ├── .env
│   └── .gitignore
│
├── [Frontend]
│   └── templates/index.html
│
└── [Docs]
    └── docs/
```

---

## Inference Layer (Online)

Các file này chạy mỗi khi có query từ user.

---

### `app.py`

Flask web server. Entry point cho production deployment.

- Khởi động Flask ngay lập tức, load pipeline trong background thread
- `GET /` → serve `templates/index.html`
- `GET /health` → polling endpoint, trả về `{ready, error, elapsed}`
- `POST /ask` → nhận query, chạy `RAGPipeline.run()`, trả về JSON
- Load `.env` qua `python-dotenv` để đọc `GROQ_API_KEY`
- Serialize numpy scalars trước khi JSON response

**Phụ thuộc vào:** `rag_inference.py`

**Run:** `python app.py [--config config.yaml] [--port 5000]`

---

### `rag_inference.py`

Orchestrator chính của toàn bộ RAG pipeline. Dùng trực tiếp khi test qua CLI.

- `RAGPipeline.__init__()`: khởi tạo `RetrievalPipeline` + `AnswerGenerator`
- `RAGPipeline.run(query)`: gọi retrieval → lấy intent → gọi generator → trả dict kết quả
- `print_result()`: in kết quả dạng bảng đẹp ra stdout
- `main()`: CLI với `--query`, `--config`, `--verbose`; lưu kết quả vào `results/last_inference.json`

**Phụ thuộc vào:** `retrieval_pipeline.py`, `generator.py`, `llm_factory.py`

**Run:** `python rag_inference.py --query "Who won the 2016 US election?"`

---

### `retrieval_pipeline.py`

Multi-stage hybrid retrieval pipeline. Thực hiện Stage 0–3 của pipeline.

| Class | Vai trò |
|---|---|
| `QueryProcessor` | Intent classification + rule-based query expansion (3 variants) |
| `HybridRetriever` | BM25 + FAISS song song, join chunk_text từ JSONL |
| `BiEncoderReranker` | 100 → 30: cosine similarity qua `models/bi_encoder` |
| `CrossEncoderReranker` | 30 → 5: full attention score qua `models/cross_encoder` |
| `RetrievalPipeline` | Orchestrator gọi 4 stage trên theo thứ tự |

- Lazy load models (chỉ load khi query đầu tiên đến)
- GPU → CPU fallback khi OOM
- In-memory embedding cache per query

**Phụ thuộc vào:** `multi_query_retriever.py`, `config.yaml`

**Run độc lập:** `python retrieval_pipeline.py --query "..."`

---

### `multi_query_retriever.py`

Helpers cho query expansion và RRF fusion. Được dùng bởi `retrieval_pipeline.py`.

| Component | Vai trò |
|---|---|
| `QueryIntentClassifier` | Rule-based intent detection (statement / biography / event / causal / opinion / general) |
| `expand_query_with_intent()` | Sinh N query variants dựa trên intent và templates tương ứng |
| `retrieve_multi_query()` | Chạy BM25 + FAISS cho từng variant, gộp bằng RRF |
| `rrf_fuse()` | Reciprocal Rank Fusion: `score = Σ 1/(k + rank_i)`, k=60 |

Không có class cấp cao — là pure functions được import vào `retrieval_pipeline.py`.

---

### `generator.py`

Answer synthesis layer. Thực hiện Stage 4–7 của pipeline.

| Class / Function | Vai trò |
|---|---|
| `ContextCompressor` | Dedup by doc_id, sort by CE, truncate 80 words, keep top-3 |
| `PromptAssembler` | Build OpenAI-format messages với `[Context N]` format và intent addendum |
| `AnswerGenerator` | Orchestrator: compress → assemble → LLM → post-process → `GenerationResult` |
| `_post_process()` | Strip preamble ("Based on the context..."), trim đến 4 câu |
| `_confidence()` | Heuristic score: `0.5×sigmoid(CE) + 0.5×keyword_coverage − penalty` |
| `_extractive_fallback()` | Lấy 2 câu đầu của chunk top-1 khi LLM fail hoàn toàn |
| `GenerationResult` | Dataclass output: `answer`, `used_docs`, `citations`, `confidence`, `fallback` |

System prompt được inject intent addendum (statement/biography/event/causal/opinion) để hướng dẫn LLM focus đúng loại thông tin.

**Phụ thuộc vào:** `groq_client.py`, `config.py`

---

### `groq_client.py`

Production Groq API client. Wraps OpenAI SDK với Groq-compatible endpoint.

- Model queue: `primary_model → fallback_models[0] → fallback_models[1]`
- Fallback chỉ trigger khi có hard failure (connection, timeout, server error)
- Auth (401/403) và quota (402) abort ngay — không retry, không fallback
- Exponential backoff + ±20% jitter; parse `try again in Xs` từ Groq rate-limit response
- Logging chi tiết: model, attempt, latency, prompt_tokens, completion_tokens
- `complete(messages)` → non-streaming, trả `str`
- `stream(messages)` → generator yield text chunks (cho SSE endpoint)

**Phụ thuộc vào:** `openai` SDK, `config.py`

---

### `llm_factory.py`

Single entry point khởi tạo `AnswerGenerator` từ `config.yaml`.

- Đọc `cfg["generator"]` → `GeneratorConfig`
- Đọc `cfg["rag_inference"]` → `RAGInferenceConfig`
- Tạo `GroqClient.from_config(gen_cfg)` → validate API key từ env
- Trả `AnswerGenerator(client, rag_cfg, min_ce_threshold)`

Tách biệt "đọc config" và "khởi tạo object" — `rag_inference.py` chỉ gọi `build_generator(cfg)` mà không cần biết chi tiết.

**Phụ thuộc vào:** `config.py`, `generator.py`, `groq_client.py`

---

## Training Layer (Offline)

Chạy một lần để tạo `models/bi_encoder` và `models/cross_encoder`.

---

### `bi_encoder_training.py`

Fine-tune Sentence-BERT trên BEIR trec-news với `MultipleNegativesRankingLoss`.

- Load corpus + queries + qrels từ `outputs/pipeline/`
- Build `(query, positive, random_negative)` triplets (≤ 100k samples)
- Train `all-MiniLM-L6-v2` với in-batch negatives, fp16, grad_accum=4
- Evaluate bằng `InformationRetrievalEvaluator` (NDCG@10, MRR@10, MAP@100)
- Save best checkpoint → `models/bi_encoder/`
- Peak VRAM: ~2.5GB trên RTX 3050 Ti

**Phụ thuộc vào:** `sentence-transformers`, `outputs/pipeline/` (data từ `run_pipeline.py`)

**Run:** `python bi_encoder_training.py [--config config.yaml]`

---

### `cross_encoder_training.py`

Fine-tune CrossEncoder trên BEIR trec-news với hard negatives từ BM25.

- Mặc định **tắt** (`enabled: false`) — pretrained `ms-marco-MiniLM-L-6-v2` đã đủ mạnh
- Build `(query, doc, label∈{0,1})` pairs; hard negatives = BM25 top-K non-relevant
- Train với `CrossEncoder.fit()` native API (tránh NaN gradient từ manual AMP)
- Evaluate bằng `CEBinaryClassificationEvaluator`
- Save → `models/cross_encoder/`
- Peak VRAM: ~2.0GB trên RTX 3050 Ti

**Phụ thuộc vào:** `sentence-transformers`, `outputs/pipeline/`, `bm25.pkl`

**Run:** `python cross_encoder_training.py [--config config.yaml]`

---

## Data Pipeline (Offline)

Chạy một lần để download, xử lý và index corpus. Output được dùng bởi cả retrieval và training.

---

### `run_pipeline.py`

CLI entry point điều phối toàn bộ offline data pipeline.

Gọi theo thứ tự:
1. `data_loader.extract_metadata()` — download và extract
2. `sampler.stratified_sample_metadata()` — sample 50% corpus
3. `splitter.split_data()` — chia train/val/test
4. `chunker.process_corpus_streaming()` — chunk thành JSONL
5. `index_builder.build_bm25_index()` — build BM25
6. `index_builder.build_faiss_index()` — build FAISS
7. `training_dataset_builder.build_training_datasets()` — tạo training data

**Run:** `python run_pipeline.py [--output_dir outputs/pipeline] [--ratio 0.5]`

---

### `data_pipeline/data_loader.py`

Download BEIR dataset từ HuggingFace, extract và save corpus/queries/qrels.

- Detect format: flattened (generated-queries) vs standard BEIR
- `extract_metadata()`: load bằng `datasets` lib, clean text, hash doc_id
- `save_sampled_data()`: ghi `corpus.jsonl`, `queries.jsonl`, `qrels.tsv`

---

### `data_pipeline/sampler.py`

Stratified sampling để giữ distribution của corpus.

- Chia corpus thành bins theo doc length (quantiles)
- Sample đều từ mỗi bin theo `ratio` (mặc định 0.5)
- Seed-controlled để reproducible

---

### `data_pipeline/splitter.py`

Split queries thành train/val/test (mặc định 80/10/10).

- Đảm bảo qrels follow query splits — không data leakage
- Output: `train_queries.jsonl`, `val_queries.jsonl`, `test_queries.jsonl` + tương ứng qrels

---

### `data_pipeline/chunker.py`

Chunk documents thành overlapping token windows cho indexing.

- `chunk_document_stream()`: sliding window với `max_tokens=256`, `overlap=32`
- Drop chunk ngắn hơn `min_tokens=30`
- Output: `corpus_chunks.jsonl` — mỗi dòng là `{chunk_id, doc_id, chunk_text, n_tokens}`
- ~1.56M chunks cho 50% BEIR trec-news

Dùng simple word tokenizer (không NLTK) để tránh memory overhead khi xử lý hàng triệu docs.

---

### `data_pipeline/index_builder.py`

Build BM25 và FAISS index từ `corpus_chunks.jsonl`.

- `build_bm25_index()`: tạo `SparseBM25`, fit trên tất cả chunk_text, save `bm25.pkl`
- `build_faiss_index()`:
  - Encode chunks bằng `SentenceTransformer` (batch_size=32, GPU nếu có)
  - Build `IndexFlatIP` (inner product = cosine sau normalize)
  - Save `faiss.index` + `chunk_metadata.pkl` (map int_pos → chunk_id/doc_id)
  - Tạo `corpus_chunks_lookup.pkl` cache để lookup nhanh chunk_text

---

### `data_pipeline/sparse_bm25.py`

Memory-efficient BM25 dùng scipy sparse matrices thay vì rank_bm25.

- `SparseBM25`: BM25Okapi với `scipy.sparse.csr_matrix` thay vì list of dicts
- Fit qua `sklearn.CountVectorizer` (identity analyzer) → token → index mapping
- `get_scores(tokenized_query)` → numpy array scores cho toàn bộ corpus
- Serializable bằng pickle — dùng cho production serving

Lý do không dùng `rank-bm25`: rank_bm25 giữ raw term counts trong memory → ~8GB+ cho 1.5M chunks.

---

### `data_pipeline/training_dataset_builder.py`

Build training datasets cho bi-encoder và cross-encoder từ split queries + qrels.

- Đọc `train_queries.jsonl` + `train_qrels.tsv`
- Bi-encoder data: `(query, positive_doc, random_negative)` triplets
- Cross-encoder data: `(query, doc, label)` pairs với label 1=relevant, 0=negative
- Output: `train_bi_encoder.jsonl`, `train_cross_encoder.jsonl`

---

### `data_pipeline/utils.py`

Shared utilities dùng trong toàn bộ data_pipeline.

- `word_tokenize_simple()`: lowercase + strip punctuation + split — lightweight tokenizer
- Helper functions xử lý text và file I/O

---

## Evaluation

---

### `evaluation.py`

Đánh giá từng stage của retrieval pipeline bằng BEIR qrels.

Evaluate 5 configurations theo thứ tự tăng dần:

| Stage | Mô tả |
|---|---|
| BM25 only | Baseline sparse retrieval |
| Dense only | FAISS với bi-encoder |
| Hybrid (BM25 + Dense + RRF) | Sau fusion |
| Hybrid + Bi-Encoder Rerank | Sau stage 2 rerank |
| Full pipeline (+ Cross-Encoder) | Final output |

Metrics: NDCG, MAP, Recall, Precision tại k ∈ {1, 3, 5, 10, 100} + MRR@10

Quy tắc nghiêm ngặt:
- Chỉ dùng qrels gốc, không simulate relevance
- Map chunk-level results lên doc-level bằng max-score aggregation
- `max_queries=500` để chạy nhanh (configurable)

**Run:** `python evaluation.py [--config config.yaml] [--max_queries 500]`

---

## Config & Infrastructure

---

### `config.py`

Typed dataclasses cho generator layer. Giúp IDE auto-complete và tránh typo khi đọc config.

| Dataclass | Chứa |
|---|---|
| `GeneratorConfig` | provider, primary_model, fallback_models, temperature, top_p, max_tokens, timeout, retry, api_key_env, base_url |
| `RAGInferenceConfig` | context_top_k, max_chunk_words, max_answer_sentences, max_context_tokens |

- `load_yaml(path)` → dict
- `get_generator_config(cfg)` → `GeneratorConfig`
- `get_rag_inference_config(cfg)` → `RAGInferenceConfig`

---

### `config.yaml`

Single source of truth cho toàn bộ hyperparameters và paths.

| Section | Kiểm soát |
|---|---|
| `paths` | Thư mục output, models, cache, results |
| `dataset` | BEIR source, HuggingFace repo, sampling ratio |
| `chunking` | max_tokens, overlap, min_tokens |
| `models` | Tên model cho bi-encoder, cross-encoder, generator |
| `generator` | Groq provider, primary/fallback models, temperature, timeout, retry |
| `indexing` | FAISS index type, embedding dim, batch size |
| `bi_encoder_training` | Learning rate, epochs, batch, amp, output path |
| `cross_encoder_training` | Learning rate, hard neg k, epochs, batch |
| `retrieval` | BM25/dense top_k, RRF k, CE threshold, query expansion |
| `evaluation` | k_values, max_queries |
| `rag_inference` | context_top_k, max_chunk_words, max_answer_sentences |

---

### `requirements.txt`

Python dependencies. Nhóm theo chức năng:

| Nhóm | Packages chính |
|---|---|
| Core ML | `torch`, `transformers`, `sentence-transformers`, `accelerate` |
| BEIR / Data | `beir`, `datasets` |
| Retrieval | `faiss-cpu`, `rank-bm25` |
| Groq API | `openai>=1.30.0`, `httpx>=0.27.0` |
| Config / Utils | `pyyaml`, `python-dotenv`, `numpy`, `pandas`, `tqdm` |
| Web Server | `flask>=3.0.0` |

---

### `.env`

API keys — **không commit, đã có trong `.gitignore`**.

```
GROQ_API_KEY=gsk_...
```

Được load bởi `app.py` qua `load_dotenv()` và đọc trong `groq_client.py` qua `os.getenv("GROQ_API_KEY")`.

---

### `.gitignore`

Exclude khỏi git:
- `.venv/` — virtual environment
- `outputs/` — FAISS index, BM25, chunks (hàng GB)
- `models/` — finetuned model weights
- `data/`, `cache/`, `checkpoints/`
- `.env` — API keys

---

## Frontend

---

### `templates/index.html`

Flask Jinja2 template cho web UI demo.

- Served bởi `GET /` trong `app.py`
- Poll `GET /health` tới khi `ready=true` rồi mới enable input
- Submit query tới `POST /ask`, hiển thị answer + metadata

---

## Docs

| File | Nội dung |
|---|---|
| `docs/pipeline.md` | RAG inference pipeline Stage 0–7, data structures, models, bottleneck |
| `docs/training.md` | Bi-encoder + cross-encoder training flow, config, VRAM, khi nào cần retrain |
| `docs/api.md` | Flask API reference: endpoints, request/response schema, curl examples |
| `docs/codebase.md` | File này — vai trò từng file trong project |
| `docs/DATA_PIPELINE_DESIGN.md` | Thiết kế offline data pipeline |
| `docs/END_TO_END_GUIDE.md` | Hướng dẫn setup từ đầu đến cuối |
| `docs/SPARSE_BM25_VS_RANK_BM25.md` | So sánh SparseBM25 và rank-bm25 về memory và tốc độ |
| `docs/sampling_methodology.md` | Phương pháp stratified sampling corpus |

---

## Dependency graph

```
config.yaml
    │
    ├── config.py ──────────────────────► groq_client.py
    │                                          │
    │                                     llm_factory.py
    │                                          │
    ├── retrieval_pipeline.py ◄── multi_query_retriever.py
    │        │
    │        └──────────────────────────► rag_inference.py ◄── app.py
    │                                          │
    │   generator.py ◄── llm_factory.py ───────┘
    │
    ├── run_pipeline.py
    │   └── data_pipeline/* (data_loader, sampler, splitter,
    │                        chunker, index_builder, sparse_bm25,
    │                        training_dataset_builder, utils)
    │
    ├── bi_encoder_training.py
    ├── cross_encoder_training.py
    └── evaluation.py
```

## Thứ tự chạy từ đầu đến cuối

```
1. python run_pipeline.py          # Download + chunk + index (một lần)
2. python bi_encoder_training.py   # Fine-tune bi-encoder (một lần)
3. python cross_encoder_training.py # Fine-tune cross-encoder (tùy chọn)
4. python evaluation.py            # Kiểm tra retrieval quality (tùy chọn)
5. python app.py                   # Production serving
   # hoặc
   python rag_inference.py --query "..."  # CLI testing
```
