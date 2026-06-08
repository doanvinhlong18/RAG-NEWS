# Training Pipeline — Bi-Encoder & Cross-Encoder

## Tổng quan

```
BEIR trec-news corpus
(~50% sampling, seed=42)
    │
    ├──► bi_encoder_training.py
    │         ▼
    │    models/bi_encoder        (finetuned SBERT)
    │
    └──► cross_encoder_training.py
              ▼
         models/cross_encoder     (finetuned CrossEncoder)
```

Cả 2 model sau khi train được dùng trực tiếp trong `retrieval_pipeline.py`.

---

## Bi-Encoder Training

**File:** `bi_encoder_training.py`

**Run:**
```bash
python bi_encoder_training.py
python bi_encoder_training.py --config config.yaml
```

### Luồng xử lý

```
corpus.jsonl + queries.jsonl + qrels.tsv
    │
    ▼  build_training_samples()
    │  - Với mỗi query, lấy tất cả positive doc_ids từ qrels
    │  - Sample random negatives từ corpus (không overlap positive)
    │  - Tạo InputExample(texts=[query, positive, negative])
    │
    ▼  List[InputExample]  (≤ max_train_samples = 100,000)
    │
    ▼  MultipleNegativesRankingLoss
    │  - In-batch negatives: các positives khác trong batch
    │    đóng vai trò negative cho query hiện tại
    │  - scale = 20.0 (temperature)
    │
    ▼  SentenceTransformer.fit()
    │  - base: sentence-transformers/all-MiniLM-L6-v2
    │  - epochs: 3
    │  - batch_size: 8, grad_accum: 4  (effective = 32)
    │  - learning_rate: 2e-5
    │  - warmup_ratio: 0.1
    │  - use_amp: true  (fp16, ~2.5GB VRAM)
    │  - max_seq_length: 256
    │  - evaluation_steps: 1000
    │
    ▼  InformationRetrievalEvaluator
    │  - NDCG@10, MRR@10, MAP@100 trên eval split (10%)
    │
    ▼  models/bi_encoder/
```

### Config (`config.yaml`)

```yaml
bi_encoder_training:
  enabled: true
  base_model: "sentence-transformers/all-MiniLM-L6-v2"
  batch_size: 8
  gradient_accumulation_steps: 4
  num_epochs: 3
  warmup_ratio: 0.1
  max_seq_length: 256
  learning_rate: 2.0e-5
  weight_decay: 0.01
  use_amp: true
  output_path: "models/bi_encoder"
  max_train_samples: 100000
  eval_split: 0.1
  evaluation_steps: 1000
```

### VRAM estimate

| Setting | VRAM |
|---|---|
| fp16, batch=8, seq=256 | ~2.5GB |
| fp32, batch=4, seq=256 | ~3.5GB |

---

## Cross-Encoder Training

**File:** `cross_encoder_training.py`

**Mặc định tắt** (`cross_encoder_training.enabled: false`) — pretrained `ms-marco-MiniLM-L-6-v2` đã đủ tốt cho news domain. Bật khi có domain-specific labeled pairs.

**Run:**
```bash
python cross_encoder_training.py
python cross_encoder_training.py --config config.yaml
```

### Luồng xử lý

```
corpus.jsonl + queries.jsonl + qrels.tsv
    │
    ▼  build_hard_negative_samples()
    │  - Lấy top-K BM25 results làm hard negatives
    │    (BM25 relevant nhưng không phải positive)
    │  - Tạo (query, doc_text, label) với label ∈ {0, 1}
    │  - Cap tại max_train_samples = 100,000
    │
    ▼  List[InputExample(texts=[query, doc], label=float)]
    │
    ▼  CrossEncoder.fit()
    │  - base: cross-encoder/ms-marco-MiniLM-L-6-v2
    │  - epochs: 3
    │  - batch_size: 4, grad_accum: 4  (effective = 16)
    │  - learning_rate: 1e-5
    │  - warmup_ratio: 0.1
    │  - use_amp: true  (fp16 native, ~2.0GB VRAM)
    │  - max_seq_length: 256
    │  - evaluation_steps: 1000
    │
    ▼  CEBinaryClassificationEvaluator
    │  - Accuracy, F1, AP trên eval split
    │
    ▼  models/cross_encoder/
```

### Config (`config.yaml`)

```yaml
cross_encoder_training:
  enabled: false
  base_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  batch_size: 4
  gradient_accumulation_steps: 4
  num_epochs: 3
  warmup_ratio: 0.1
  max_seq_length: 256
  learning_rate: 1.0e-5
  hard_neg_top_k: 20
  use_amp: true
  output_path: "models/cross_encoder"
  max_train_samples: 100000
  evaluation_steps: 1000
```

### VRAM estimate

| Setting | VRAM |
|---|---|
| fp16, batch=4, seq=256 | ~2.0GB |

---

## Khi nào cần train lại

| Trigger | Model cần train lại |
|---|---|
| Thêm domain mới vào corpus | Cả 2 |
| Retrieval recall thấp | Bi-Encoder |
| Rerank precision thấp | Cross-Encoder |
| Thay đổi chunking strategy | Cả 2 |

---

## Dependency giữa 2 model

Cross-Encoder training dùng **BM25 hard negatives** — nên chạy sau khi đã build BM25 index.

```
build_index (BM25) → cross_encoder_training → models/cross_encoder
build_index (FAISS) + bi_encoder_training  → models/bi_encoder
```

Cả 2 model **độc lập** với nhau — có thể train song song nếu đủ VRAM.

---

## Files liên quan

| File | Vai trò |
|---|---|
| `bi_encoder_training.py` | Fine-tune SBERT với MultipleNegativesRankingLoss |
| `cross_encoder_training.py` | Fine-tune CrossEncoder với hard negatives từ BM25 |
| `config.yaml` | `bi_encoder_training` + `cross_encoder_training` sections |
| `models/bi_encoder/` | Output: finetuned SBERT (dùng trong retrieval + rerank) |
| `models/cross_encoder/` | Output: finetuned CrossEncoder (dùng trong CE rerank) |
