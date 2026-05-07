# RAG Pipeline — Luồng xử lý từ Query đến Answer

## Tổng quan

```
Query (string)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 0 — Query Expansion                         │
│  "Who won 2016 election?"                           │
│  → ["Who won 2016 election?",                       │
│      "Who won 2016 election? causes",               │
│      "Who won 2016 election? impact"]               │
└──────────────────────┬──────────────────────────────┘
                       │ 3 query variants
                       ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 1 — Hybrid Retrieval                         │
│                                                     │
│  BM25 (sparse)   ──┐                                │
│  top-100 chunks    ├──► RRF Fusion ──► top-100      │
│  FAISS (dense)   ──┘   (per query,                  │
│  top-100 chunks        then across queries)         │
└──────────────────────┬──────────────────────────────┘
                       │ 100 candidates (chunk_id + rrf_score)
                       │ + join chunk_text từ corpus_chunks.jsonl
                       ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 2 — Bi-Encoder Rerank                        │
│  100 → 30                                           │
│  cosine(query_emb, doc_emb) — models/bi_encoder     │
└──────────────────────┬──────────────────────────────┘
                       │ 30 candidates + bi_encoder_score
                       ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 3 — Cross-Encoder Rerank                     │
│  30 → 5                                             │
│  score(query, doc_text) — models/cross_encoder      │
└──────────────────────┬──────────────────────────────┘
                       │ 5 docs + ce_score
                       ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 4 — Context Compression                      │
│  - Lọc chunk_text rỗng                              │
│  - Dedup theo doc_id (giữ CE cao nhất)              │
│  - Sort theo CE score descending                    │
│  - Truncate mỗi chunk ≤ 80 words                    │
│  - Giữ top-3 chunks                                 │
└──────────────────────┬──────────────────────────────┘
                       │ top-3 compressed docs
                       ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 5 — Prompt Assembly                          │
│  SYSTEM: intent-aware synthesis rules               │
│  USER:   [Context 1] / [Context 2] / [Context 3]   │
│          + User Question                            │
└──────────────────────┬──────────────────────────────┘
                       │ OpenAI-format messages list
                       ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 6 — Groq LLM Generation                     │
│  Primary:  llama-3.3-70b-versatile                  │
│  Fallback: qwen-qwq-32b → mixtral-8x7b-32768        │
│  Retry: 3 attempts, exponential backoff + jitter    │
└──────────────────────┬──────────────────────────────┘
                       │ raw answer string
                       ▼
┌─────────────────────────────────────────────────────┐
│  STAGE 7 — Post-processing                          │
│  - Strip "Based on the context..." preamble         │
│  - Trim to max 4 sentences                          │
│  - Heuristic confidence score                       │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
            GenerationResult {
              answer, used_docs,
              citations, confidence,
              fallback
            }
```

---

## Chi tiết từng stage

### Stage 0 — Query Expansion

**Files:** `multi_query_retriever.py`, `retrieval_pipeline.py` (`QueryProcessor`)

Phân loại intent của query trước (statement / biography / event / causal / opinion / general), sau đó sinh ra `n_variants` biến thể (mặc định 3) bằng rule-based templates tương ứng với intent. T5 rewriter bị tắt (`use_t5_rewriter: false`) để tiết kiệm VRAM.

**Templates ví dụ (intent = event):**

```
"{query}"
"What happened: {query}"
"{query} timeline"
```

**Mục đích:** tăng recall — mỗi biến thể có thể match các từ khác nhau trong corpus.

---

### Stage 1 — Hybrid Retrieval

**File:** `retrieval_pipeline.py` (`HybridRetriever`)

Chạy song song 2 hệ thống tìm kiếm cho từng query variant:

| Phương pháp | Loại | Thế mạnh |
|---|---|---|
| **BM25** (`SparseBM25`) | Sparse / keyword | Proper nouns, dates, từ chính xác |
| **FAISS** (`IndexFlatIP`) | Dense / semantic | Ngữ nghĩa, paraphrase, synonyms |

Kết quả 6 danh sách (3 queries × 2 methods) được gộp bằng **RRF (Reciprocal Rank Fusion)** theo 2 bước:

1. RRF per-query: gộp BM25 + FAISS của từng query
2. RRF across queries: gộp kết quả của 3 query variants

```
RRF score = Σ  1 / (k + rank_i)     k = 60 (config)
```

RRF dùng **rank** thay vì raw score → robust với sự khác biệt scale giữa BM25 và cosine similarity.

Sau bước này, kết quả được join với `corpus_chunks.jsonl` để lấy `chunk_text`.

---

### Stage 2 — Bi-Encoder Rerank

**File:** `retrieval_pipeline.py` (`BiEncoderReranker`)

**100 → 30 candidates**

Re-score bằng `models/bi_encoder` (finetuned trên BEIR trec-news). Encode query và docs **độc lập** → cosine similarity.

```
score = cosine(encode(query), encode(doc))
```

**Ưu điểm:** nhanh (encode riêng, batch được).
**Nhược điểm:** kém chính xác hơn CE vì không có full attention giữa query và doc.

---

### Stage 3 — Cross-Encoder Rerank

**File:** `retrieval_pipeline.py` (`CrossEncoderReranker`)

**30 → 5 candidates**

Score từng cặp `(query, doc_text)` bằng `models/cross_encoder` (finetuned). Query và doc được encode **cùng nhau** qua full transformer attention.

```
score = CE(query [SEP] doc_text)
```

**Ưu điểm:** chính xác nhất, quyết định ranking cuối cùng.
**Nhược điểm:** không scale được cho tập lớn (O(n) forward passes).

Áp dụng `ce_score_threshold = 0` (config) để lọc docs hoàn toàn không liên quan.

---

### Stage 4 — Context Compression

**File:** `generator.py` (`ContextCompressor`)

Xử lý 5 docs trước khi đưa vào prompt:

1. **Lọc** chunk có `chunk_text` rỗng hoặc chỉ có whitespace
2. **Dedup** theo `doc_id` — nếu nhiều chunk cùng doc, giữ chunk có CE score cao nhất
3. **Sort** theo CE score descending
4. **Truncate** mỗi chunk xuống ≤ 80 words để tránh overflow token budget
5. **Slice** giữ top-3 chunks (config: `context_top_k = 3`)

Mục đích: giảm context stuffing, tối ưu token usage, tăng precision của LLM.

---

### Stage 5 — Prompt Assembly

**File:** `generator.py` (`PromptAssembler`)

Tạo OpenAI-style messages với system prompt intent-aware:

```
[
  {
    "role": "system",
    "content": "You are a concise retrieval-augmented AI assistant.
                Synthesize information from retrieved context.
                Answer directly. Do NOT open with 'Based on the context...'.
                Keep answer short (2-4 sentences), factual.
                If context is insufficient: 'I do not have enough reliable information to answer that.'"
  },
  {
    "role": "user",
    "content": "[Context 1]
                <chunk_text_1>

                [Context 2]
                <chunk_text_2>

                [Context 3]
                <chunk_text_3>

                User Question:
                <query>"
  }
]
```

Intent addendum được append vào system prompt cho các loại query đặc biệt (statement / biography / event / causal / opinion).

---

### Stage 6 — Groq LLM Generation

**File:** `groq_client.py` (`GroqClient`)

Sử dụng OpenAI SDK với Groq-compatible endpoint:

```
Base URL: https://api.groq.com/openai/v1
API Key:  GROQ_API_KEY (từ .env)
```

**Model queue (fallback tự động):**

| Priority | Model | Ghi chú |
|---|---|---|
| Primary | `llama-3.3-70b-versatile` | Best quality |
| Fallback 1 | `qwen-qwq-32b` | Reasoning-focused |
| Fallback 2 | `mixtral-8x7b-32768` | Fast, long context |

**Generation settings:** `temperature=0.1`, `top_p=0.9`, `max_tokens=256`

**Error handling:**

| Error | Xử lý |
|---|---|
| `429 rate_limit` | Exponential backoff + jitter, parse `try again in Xs` |
| `401/403 auth` | Abort ngay, không retry |
| `402 quota` | Abort ngay, không retry |
| `timeout` | Retry tối đa 3 lần |
| `connection` | Retry, fallback model |
| LLM hoàn toàn fail | Extractive fallback (câu đầu của chunk top-1) |

---

### Stage 7 — Post-processing

**File:** `generator.py` (`_post_process`, `_confidence`)

#### Preamble Removal

Xóa các opener không cần thiết mà LLM thường sinh ra:

```
"Based on the context, ..."       → bỏ
"According to the documents, ..." → bỏ
"The context states that ..."     → bỏ
```

#### Sentence Trimming

Giữ tối đa `max_answer_sentences = 4` câu đầu tiên.

#### Confidence Score

Heuristic kết hợp 3 thành phần:

```
confidence = 0.5 × sigmoid(avg_CE) + 0.5 × keyword_coverage − dont_know_penalty
```

- `sigmoid(avg_CE)`: normalise CE score về [0, 1]
- `keyword_coverage`: % từ trong answer xuất hiện trong context
- `dont_know_penalty = 0.6` nếu answer chứa "do not have enough"

---

## Cấu trúc dữ liệu

### Offline (build một lần)

```
Corpus gốc
(BEIR trec-news, ~50% sampling, seed=42)
    │
    ▼  chunker.py
    │  max_tokens=256, overlap=32
    ▼
corpus_chunks.jsonl          ← 1,559,239 chunks
{chunk_id, doc_id, chunk_text, chunk_tokens, n_tokens}
    │
    ├──► index_builder.py (BM25)
    │         ▼
    │    bm25.pkl              ← SparseBM25 (sklearn CountVectorizer)
    │
    └──► index_builder.py (FAISS)
              ▼
         faiss.index           ← IndexFlatIP, 1.5M × 384-dim float32
         chunk_metadata.pkl    ← {int_pos: {chunk_id, doc_id}}
         corpus_chunks_lookup.pkl  ← {chunk_id: {chunk_text, title}}  (cache)
```

### Online (mỗi query)

```
Query string
    → List[str] (3 variants)
    → List[Tuple[chunk_id, rrf_score]] (100 items)
    → List[Dict] (100 items, enriched với chunk_text)
    → List[Dict] (30 items, + bi_encoder_score)
    → List[Dict] (5 items, + ce_score)
    → List[Dict] (3 items, compressed/truncated 80 words)
    → GenerationResult {answer, used_docs, citations, confidence, fallback}
```

---

## Models đang sử dụng

| Role | Model | Finetuned? | Vị trí |
|---|---|---|---|
| Dense retrieval (FAISS encode) | `sentence-transformers/all-MiniLM-L6-v2` | Có | `models/bi_encoder` |
| Bi-encoder rerank | `models/bi_encoder` | Có | shared với FAISS |
| Cross-encoder rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Có | `models/cross_encoder` |
| Answer generation | `llama-3.3-70b-versatile` | Không (API) | Groq cloud |

---

## Bottleneck & thời gian chạy

| Giai đoạn | Thời gian | Loại |
|---|---|---|
| FAISS index load | ~5s | One-time cold start |
| BM25 load | ~8s | One-time cold start |
| Chunk lookup load (lần 1) | ~29s | One-time, build pickle cache |
| Chunk lookup load (lần 2+) | ~4s | Đọc từ `_lookup.pkl` |
| FAISS search (3 queries) | ~4s | Per-query, linear scan 1.5M vectors |
| Bi-encoder rerank | ~0.5s | Per-query, GPU |
| Cross-encoder rerank | ~0.3s | Per-query, GPU |
| Groq API call | ~0.5–2s | Per-query, network |
| **Tổng (sau warm-up)** | **~5–7s** | Per-query |

---

## Files liên quan

| File | Vai trò |
|---|---|
| `rag_inference.py` | Entry point CLI + `RAGPipeline` orchestrator |
| `retrieval_pipeline.py` | `RetrievalPipeline`: Stage 0–3 |
| `multi_query_retriever.py` | Query expansion, intent classification, RRF fusion |
| `generator.py` | `AnswerGenerator`: Stage 4–7 |
| `groq_client.py` | Groq API client: retry, fallback, streaming |
| `llm_factory.py` | Factory: config → `GroqClient` + `AnswerGenerator` |
| `config.py` | Typed dataclasses cho generator + RAG inference config |
| `config.yaml` | Tất cả hyperparameters và paths |
| `app.py` | Flask web server |
