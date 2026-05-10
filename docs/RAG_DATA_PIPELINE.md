# RAG Data Pipeline — Tài liệu kỹ thuật

> **Mục tiêu:** Xây dựng một bộ dữ liệu huấn luyện chất lượng cao cho hệ thống RAG (Retrieval-Augmented Generation) từ các dataset công khai trên HuggingFace, với thiết kế ưu tiên tối ưu bộ nhớ (memory-safe) và khả năng mở rộng.

---

## Tổng quan kiến trúc

```
HuggingFace Dataset
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 1: Extract Metadata (Streaming Pass 1)           │
│  Lọc outlier, dedup, thu thập thống kê phân phối        │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 2: Stratified Sampling                           │
│  Chọn mẫu đại diện, giữ balance phân phối gốc          │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 3: Save Sampled Data (Streaming Pass 2)          │
│  Ghi corpus, queries, qrels ra disk                     │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 4: Train / Val / Test Split                      │
│  Chia theo query, tránh data leakage                    │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 5: Chunking & Tokenization (Streaming)           │
│  Chia document thành các chunk có overlap               │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 6A: BM25 Index           STAGE 6B: FAISS Index   │
│  SparseBM25 (scipy)             SentenceTransformer +   │
│  Sparse lexical retrieval       Dense vector retrieval  │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  STAGE 7: Training Dataset Builder                      │
│  Tạo (query, positive, negatives) cho Bi/Cross Encoder  │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
         train.jsonl / val.jsonl / test.jsonl
```

---

## Chi tiết từng bước

### Stage 1 — Extract Metadata (Streaming Pass 1)
**File:** `data_loader.py` → `extract_metadata()`

**Làm gì:**
Streaming qua toàn bộ dataset HuggingFace mà **không load vào RAM**. Chỉ trích xuất metadata tối thiểu (doc_id, word_count) để phục vụ bước sampling.

**Tại sao cần:**
Các dataset BEIR như MS-MARCO hay Natural Questions có hàng triệu tài liệu — không thể load toàn bộ text vào memory. Chỉ cần metadata là đủ để quyết định lấy doc nào.

**Xử lý trong bước này:**

| Bộ lọc | Mô tả | Lý do |
|--------|-------|-------|
| **Length filter** | Loại doc có `word_count < min_length` hoặc `> max_length` | Tránh doc quá ngắn (vô nghĩa) hoặc quá dài (gây OOM khi encode) |
| **Deduplication** | Hash MD5 của cleaned text, bỏ các bản trùng | Dataset công khai thường có nhiều doc duplicate |
| **Text cleaning** | Lowercase, xóa ký tự đặc biệt, chuẩn hóa khoảng trắng | Đảm bảo hash dedup chính xác, giảm noise |

**Hỗ trợ 2 schema dataset:**
- **Standard BEIR:** Có split riêng `corpus`, `queries`, `qrels`
- **Flattened:** Dataset được flatten thành 1 split `train` duy nhất (dạng `generated-queries`)

**Output:** Ba dictionary trong memory: `corpus_meta`, `queries_meta`, `qrels_dict` — chỉ lưu ID và độ dài, không lưu text.

---

### Stage 2 — Stratified Sampling
**File:** `sampler.py` → `stratified_sample_metadata()`

**Làm gì:**
Chọn một tập con (theo `ratio`) từ toàn bộ corpus và queries, đảm bảo **phân phối của tập con phản ánh phân phối gốc**.

**Tại sao cần:**
- Random sampling thuần túy có thể bỏ sót các loại document hiếm (ví dụ: doc rất ngắn hoặc rất dài)
- Cần đảm bảo mỗi query trong tập sample có đủ relevant documents đi kèm

**Chiến lược sampling:**

**Với corpus:**
Chia doc vào 3 bin theo độ dài (dùng quantile), sample đều từ từng bin để giữ distribution.

**Với queries:**
Stratify theo 2 chiều:
1. **Số lượng qrels** (query có nhiều hay ít relevant doc)
2. **Độ dài trung bình của relevant docs**

→ Tạo ma trận stratum `(bin_qrels × bin_doc_len)`, sample `ratio` từ mỗi ô.

**Ưu tiên relevant docs:**
Sau khi chọn queries, tự động đưa tất cả relevant doc của các query đó vào corpus. Sau đó mới fill thêm doc ngẫu nhiên (theo bin) để đạt đủ `ratio`.

**Output:** `sampled_qids: Set[str]`, `sampled_doc_ids: Set[str]`

---

### Stage 3 — Save Sampled Data (Streaming Pass 2)
**File:** `data_loader.py` → `save_sampled_data()`

**Làm gì:**
Stream lại dataset lần 2, lần này **chỉ ghi ra disk** những record thuộc tập sample. Áp dụng cleaning trước khi ghi.

**Tại sao cần pass 2 riêng biệt:**
Nếu gộp pass 1 và pass 2, ta phải giữ text của tất cả candidate docs trong memory trong khi đang đọc → OOM. Tách thành 2 pass cho phép pass 1 chỉ giữ metadata (nhẹ), pass 2 xử lý và discard text ngay sau khi ghi.

**Output files:**
```
outputs/
├── sampled_corpus.jsonl      # {doc_id, title, text}
├── sampled_queries.json      # {qid: query_text}
├── sampled_qrels.json        # {qid: {doc_id: score}}
└── pipeline_stats.json       # Thống kê tỉ lệ sample
```

---

### Stage 4 — Train / Val / Test Split
**File:** `splitter.py` → `split_data()`

**Làm gì:**
Shuffle ngẫu nhiên danh sách queries (với fixed seed), sau đó chia theo tỉ lệ `train:val:test` (mặc định `0.8:0.1:0.1`).

**Tại sao quan trọng:**
Split phải thực hiện **trên queries**, không phải trên documents. Nếu split trên doc, cùng 1 doc có thể xuất hiện ở cả train và test → **data leakage** → metric eval bị inflate.

Qrels theo query sang split tương ứng, corpus file giữ nguyên (toàn bộ split đều dùng chung).

**Output files:**
```
outputs/
├── train_queries.json / train_qrels.json
├── val_queries.json   / val_qrels.json
└── test_queries.json  / test_qrels.json
```

---

### Stage 5 — Chunking & Tokenization
**File:** `chunker.py` → `process_corpus_streaming()`

**Làm gì:**
Đọc `sampled_corpus.jsonl` dòng-by-dòng, chia mỗi document thành các chunk có kích thước cố định với overlap.

**Tại sao cần chunking:**
- Encoder models (FAISS) có giới hạn độ dài input (thường 256–512 tokens)
- BM25 hoạt động tốt hơn trên đoạn văn tập trung hơn là toàn bộ document dài
- Retrieval chunk-level → map về document-level khi trả kết quả

**Tham số:**

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `max_tokens` | 256 | Số token tối đa mỗi chunk |
| `overlap` | 32 | Số token overlap giữa 2 chunk liền kề |

**Lý do cần overlap:**
Tránh trường hợp câu trả lời bị cắt đứt ở ranh giới 2 chunk — overlap đảm bảo thông tin ngữ cảnh không bị mất.

**Tokenizer:** Sử dụng `word_tokenize_simple()` (regex-based) thay vì NLTK để tránh load model nặng khi xử lý hàng triệu doc.

**Chunk ID format:** `{doc_id}__chunk_{idx}` — cho phép map ngược về doc gốc.

**Output:** `corpus_chunks.jsonl` với mỗi dòng gồm `chunk_id`, `doc_id`, `chunk_text`, `n_tokens`, `chunk_tokens`.

---

### Stage 6A — BM25 Index (Sparse Lexical)
**File:** `index_builder.py` → `build_bm25_index()`  
**Implementation:** `sparse_bm25.py` → `SparseBM25`

**Làm gì:**
Xây dựng BM25 index từ tokenized chunks, lưu dưới dạng scipy sparse matrix.

**Tại sao không dùng `rank_bm25` library thông thường:**
`rank_bm25` lưu toàn bộ corpus dưới dạng list of dictionaries trong RAM → với 500k+ chunks sẽ OOM. `SparseBM25` tùy chỉnh dùng `scipy.sparse.csc_matrix` để:
- Chỉ lưu non-zero term frequencies (memory hiệu quả hơn 10–100×)
- Hỗ trợ streaming fit qua generator, không cần load toàn bộ vào memory
- Serializable qua pickle

**Công thức BM25 Okapi:**

```
score(q, d) = Σ IDF(t) × [tf(t,d) × (k1 + 1)] / [tf(t,d) + k1 × (1 - b + b × |d|/avgdl)]
```

**Tham số mặc định:** `k1=1.5`, `b=0.75`, `epsilon=0.25`

**Output:** `bm25.pkl` — serialized `SparseBM25` object.

---

### Stage 6B — FAISS Index (Dense Semantic)
**File:** `index_builder.py` → `build_faiss_index()`

**Làm gì:**
Encode từng chunk thành dense vector bằng SentenceTransformer, add vào FAISS `IndexFlatIP` (inner product = cosine similarity với normalized vectors).

**Tại sao dùng batched streaming:**
Không thể encode toàn bộ corpus cùng lúc (N × 384 float32 với N=500k sẽ chiếm ~750MB chỉ riêng embeddings). Streaming theo batch cho phép kiểm soát memory footprint.

**GPU optimization:**
- Sử dụng `model.half()` (float16) trên CUDA để tăng throughput ~2×
- Tự động fallback về CPU với batch nhỏ hơn nếu OOM
- Vectors được normalize trước khi add → dùng `IndexFlatIP` tương đương cosine

**Output files:**
```
outputs/
├── faiss.index          # FAISS binary index
└── chunk_metadata.pkl   # {faiss_idx: {chunk_id, doc_id}}
```

---

### Stage 7 — Training Dataset Builder
**File:** `training_dataset_builder.py` → `build_training_datasets()`

**Làm gì:**
Tạo training pairs dưới dạng `(query, positive_docs, negative_docs)` cho từng split.

**Format output (JSONL):**
```json
{
  "query": "what causes inflation",
  "positives": ["Inflation is caused by..."],
  "negatives": ["The weather today is..."]
}
```

**Negative sampling strategy:**
Dùng **random negatives** (lấy doc ngẫu nhiên từ corpus, loại trừ positive). Đây là lựa chọn có chủ đích:
- BM25 hard negatives cần query BM25 index cho mỗi query → chậm và tốn memory
- Random negatives đủ tốt cho giai đoạn pre-training bi-encoder
- Hard negatives nên được thêm vào ở giai đoạn fine-tuning sau với mining riêng

**2-pass approach để tránh OOM:**
1. **Pass 1:** Generate tất cả `(qid, pos_ids, neg_ids)` pair — chỉ lưu IDs
2. **Pass 2:** Stream corpus, chỉ load text của các doc ID cần thiết (`needed_dids`)

**Output:** `train.jsonl`, `val.jsonl`, `test.jsonl`

---

## Cách chạy pipeline

### Yêu cầu
```bash
pip install datasets sentence-transformers faiss-cpu scipy scikit-learn torch tqdm
```

### Chạy toàn bộ pipeline
```bash
python run_pipeline.py \
  --repo_id "BeIR/trec-news" \
  --dataset_name "trec-news" \
  --output_dir "outputs/trec-news" \
  --ratio 0.5 \
  --seed 42 \
  --model_name "sentence-transformers/all-MiniLM-L6-v2"
```

### Tham số CLI

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--repo_id` | `BeIR/trec-news` | HuggingFace dataset repo ID |
| `--dataset_name` | `trec-news` | Config name của dataset |
| `--output_dir` | `outputs/pipeline` | Thư mục output |
| `--ratio` | `0.5` | Tỉ lệ sample (0.0–1.0) |
| `--seed` | `42` | Random seed để reproducibility |
| `--model_name` | `all-MiniLM-L6-v2` | SentenceTransformer model cho FAISS |

### Idempotency
Pipeline **tự động skip** các bước đã hoàn thành (kiểm tra file output tồn tại). Nếu pipeline bị ngắt giữa chừng, chạy lại sẽ tiếp tục từ bước còn lại.

---

## Cấu trúc output cuối cùng

```
outputs/pipeline/
├── sampled_corpus.jsonl          # Corpus đã sample & clean
├── sampled_queries.json          # Queries đã sample
├── sampled_qrels.json            # Relevance judgments
├── pipeline_stats.json           # Thống kê pipeline
│
├── train_queries.json            # Queries tập train
├── train_qrels.json              # Qrels tập train
├── val_queries.json              # Queries tập val
├── val_qrels.json                # Qrels tập val
├── test_queries.json             # Queries tập test
├── test_qrels.json               # Qrels tập test
│
├── corpus_chunks.jsonl           # Chunks + tokens (cho BM25)
│
├── bm25.pkl                      # BM25 sparse index
├── faiss.index                   # FAISS dense index
├── chunk_metadata.pkl            # Mapping FAISS idx → chunk/doc ID
│
├── train.jsonl                   # Training data (query, pos, neg)
├── val.jsonl                     # Validation data
└── test.jsonl                    # Test data
```

---

## Nguyên tắc thiết kế

### Memory-safe by design
Mỗi bước đều dùng **streaming** — không có bước nào load toàn bộ corpus vào RAM cùng lúc. Pipeline này có thể xử lý dataset hàng chục triệu doc trên máy 16GB RAM.

### Reproducibility
Fixed `seed` được truyền xuyên suốt pipeline. Với cùng `seed` và `ratio`, output sẽ hoàn toàn giống nhau.

### No data leakage
Split được thực hiện trên **queries**, không phải documents. Corpus file dùng chung cho tất cả splits nhưng queries và qrels được tách hoàn toàn.

### Extensibility
- Thêm dataset mới: chỉ cần implement thêm schema handler trong `data_loader.py`
- Thay encoder: truyền `--model_name` khác, pipeline tự build lại FAISS
- Thêm hard negatives: extend `training_dataset_builder.py` với BM25 mining post-hoc

---

## Các lưu ý quan trọng

> **`chunk_tokens` trong `corpus_chunks.jsonl`:** Field này được giữ lại để build BM25 index mà không cần tokenize lại. Sau khi BM25 được build xong, nếu muốn tiết kiệm disk có thể strip field này.

> **Flattened vs Standard schema:** Dataset dạng `generated-queries` thường flatten corpus và queries vào cùng 1 split. Pipeline tự detect qua `is_flattened()` dựa trên tên repo/dataset.

> **FAISS `IndexFlatIP` vs `IndexIVFFlat`:** Pipeline dùng `IndexFlatIP` (brute-force exact search). Với corpus >1M chunks, cân nhắc chuyển sang `IndexIVFFlat` hoặc `IndexHNSW` để trade accuracy cho speed.
