# Flask API Reference

## Khởi động server

```bash
python app.py
python app.py --config config.yaml --port 5000 --host 0.0.0.0
```

Server khởi động ngay, pipeline load trong background thread. Frontend poll `/health` để biết khi nào ready.

---

## Endpoints

### `GET /`

Trả về giao diện web (`templates/index.html`).

---

### `GET /health`

Kiểm tra trạng thái pipeline.

**Response:**

```json
{
  "ready":   true,
  "error":   null,
  "elapsed": 42.3
}
```

| Field | Type | Mô tả |
|---|---|---|
| `ready` | bool | Pipeline đã load xong chưa |
| `error` | string \| null | Error message nếu load thất bại |
| `elapsed` | float | Giây đã trôi qua kể từ khi bắt đầu load |

**Trạng thái có thể:**

| `ready` | `error` | Ý nghĩa |
|---|---|---|
| `false` | `null` | Đang load |
| `true` | `null` | Sẵn sàng |
| `false` | `"..."` | Load thất bại |

---

### `POST /ask`

Gửi query, nhận answer từ RAG pipeline.

**Request body (JSON):**

```json
{
  "query": "Who won the 2016 US election?"
}
```

| Field | Type | Required | Constraint |
|---|---|---|---|
| `query` | string | Có | 1–1000 ký tự |

**Response thành công (200):**

```json
{
  "query":      "Who won the 2016 US election?",
  "intent":     "event",
  "answer":     "Donald Trump won the 2016 US presidential election.",
  "used_docs":  ["a80440a0", "b12345ff"],
  "citations":  [],
  "confidence": 0.7832,
  "fallback":   false,
  "latency_s":  1.42,
  "retrieved": [
    {
      "doc_id":     "a80440a0",
      "chunk_id":   "a80440a0_0",
      "chunk_text": "Donald Trump defeated Hillary Clinton...",
      "ce_score":   4.2183,
      "rrf_score":  0.0312
    }
  ]
}
```

| Field | Type | Mô tả |
|---|---|---|
| `query` | string | Query gốc |
| `intent` | string | Intent được classify: `general` / `event` / `statement` / `biography` / `causal` / `opinion` |
| `answer` | string | Answer tổng hợp từ LLM |
| `used_docs` | List[string] | doc_id của các chunks đưa vào LLM |
| `citations` | List[string] | doc_id được LLM cite trong answer |
| `confidence` | float | Heuristic score [0, 1] |
| `fallback` | bool | `true` nếu LLM fail → dùng extractive fallback |
| `latency_s` | float | Tổng thời gian xử lý (giây) |
| `retrieved` | List[Dict] | Danh sách docs reranked (raw, trước context compression) |

**Error responses:**

| HTTP | Body | Nguyên nhân |
|---|---|---|
| `400` | `{"error": "Query cannot be empty."}` | Query rỗng |
| `400` | `{"error": "Query too long (max 1000 chars)."}` | Query > 1000 ký tự |
| `500` | `{"error": "<message>"}` | Pipeline exception |
| `503` | `{"error": "Pipeline is still loading. Please wait."}` | Pipeline chưa ready |

---

## Ví dụ curl

```bash
# Kiểm tra trạng thái
curl http://localhost:5000/health

# Gửi query
curl -X POST http://localhost:5000/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "Who won the 2016 US election?"}'
```

---

## Pipeline lifecycle

```
app.py starts
    │
    ├── Flask server: ready ngay lập tức
    │
    └── Background thread: _init_pipeline()
            │
            ├── Load RetrievalPipeline
            │   (FAISS index, BM25, bi-encoder, cross-encoder)
            │   ~45s lần đầu, ~15s lần 2+
            │
            └── Build AnswerGenerator
                (GroqClient from .env GROQ_API_KEY)
                ~instant

/health → {"ready": false} trong lúc load
/health → {"ready": true}  sau khi xong
/ask   → 503 trong lúc load
/ask   → 200 sau khi xong
```

---

## Environment variables

| Variable | Nguồn | Bắt buộc |
|---|---|---|
| `GROQ_API_KEY` | `.env` | Có — nếu thiếu pipeline init sẽ fail |
| `HF_HUB_DISABLE_SYMLINKS_WARNING` | `app.py` (auto-set) | Không |
