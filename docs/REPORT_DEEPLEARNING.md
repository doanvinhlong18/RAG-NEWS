# Báo cáo môn học Deep Learning
## Xây dựng hệ thống Retrieval-Augmented Generation cho Trả lời Câu hỏi Tin tức

---

## 1. Tóm tắt (Abstract)

Báo cáo trình bày việc xây dựng và đánh giá một hệ thống **Retrieval-Augmented Generation (RAG)** phục vụ bài toán trả lời câu hỏi dựa trên kho tin tức tiếng Anh (BEIR trec-news). Hệ thống triển khai pipeline truy xuất đa giai đoạn gồm BM25, FAISS dense retrieval, Reciprocal Rank Fusion (RRF), bi-encoder reranking và cross-encoder reranking, kết thúc bằng tổng hợp câu trả lời qua LLM (Groq llama-3.3-70b-versatile).

Ba cấu hình được so sánh để đánh giá ảnh hưởng của fine-tuning:
- **(C1)** Không fine-tune (base models)
- **(C2)** Chỉ fine-tune Bi-Encoder
- **(C3)** Fine-tune cả Bi-Encoder và Cross-Encoder

Kết quả nổi bật: pretrained cross-encoder (ms-marco-MiniLM-L-6-v2) chuyển giao rất tốt sang domain tin tức (NDCG@10 = **0.4948**), trong khi fine-tuning thêm CE trên dữ liệu BEIR trec-news với hard negatives từ BM25 lại gây suy giảm mạnh (NDCG@10 giảm xuống còn **0.3156**) — một finding thực nghiệm quan trọng về transfer learning và overfitting trong reranking.

---

## 2. Giới thiệu và Động lực

### 2.1 Bài toán

Cho trước một kho tài liệu tin tức và câu hỏi tự nhiên, hệ thống cần:
1. Truy xuất các đoạn văn liên quan từ kho ~1.56M chunks (~594K tài liệu gốc, lấy mẫu 50%)
2. Tổng hợp câu trả lời ngắn gọn, có căn cứ hoàn toàn từ ngữ cảnh truy xuất

### 2.2 Thách thức

| Thách thức | Mô tả |
|---|---|
| Kho dữ liệu lớn | 1.56M chunks, FAISS index ~2.4 GB |
| Lexical gap | Câu hỏi ngắn, tài liệu dài — BM25 đơn thuần bỏ sót nhiều |
| Tài nguyên hạn chế | RTX 3050 Ti 4 GB VRAM — phải tối ưu memory |
| Đánh giá end-to-end | Cần đánh giá từng giai đoạn riêng biệt (BEIR format) |

### 2.3 Đóng góp chính

- Pipeline RAG 6 giai đoạn hoàn chỉnh, chạy trên GPU consumer-grade 4 GB VRAM
- Phân tích so sánh hệ thống **3 cấu hình fine-tuning** trên BEIR trec-news
- Phát hiện thực nghiệm: **fine-tuning cross-encoder với BM25 hard negatives gây hại** khi pretrained model đã transfer tốt sang domain mới

---

## 3. Kiến trúc Hệ thống

```
Câu hỏi người dùng
      ↓
Query Expansion (n=3 variants, intent-aware)
      ↓
Hybrid Retrieval: BM25 top-100  +  FAISS dense top-100
      ↓
RRF Fusion per variant → cross-variant fusion (top-100)
      ↓
Bi-Encoder Rerank (100 → 30)
      ↓
Cross-Encoder Rerank (30 → 5)
      ↓
Context Compression (dedup by doc, truncate 150 words)
      ↓
Groq LLM (llama-3.3-70b-versatile) → Câu trả lời
```

### 3.1 Các thành phần chính

**BM25 (Sparse Retrieval)**
- Triển khai tùy chỉnh `SparseBM25` dùng scipy sparse matrix thay cho rank-bm25 để tiết kiệm RAM
- Tokenization: lowercase + remove punctuation + remove stopwords

**FAISS Dense Retrieval**
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dim) → fine-tune thành `models/bi_encoder`
- Index: `IndexFlatIP` (cosine similarity via L2-normalized vectors)
- Encoding: fp16 để giảm VRAM

**Reciprocal Rank Fusion**
- Công thức: `score(d) = Σ 1 / (k + rank(d))`, k = 60
- Fuse BM25 + Dense per query variant, sau đó fuse across variants

**Bi-Encoder Reranking**
- Dùng lại `models/bi_encoder`, encode query và 100 candidate chunks, tính cosine similarity
- Pre-encode toàn bộ query trong batch để tối ưu tốc độ

**Cross-Encoder Reranking**
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` → fine-tune thành `models/cross_encoder`
- Full attention giữa (query, passage) — chính xác hơn bi-encoder nhưng không scale được

**Query Expansion**
- Phân loại intent: event / statement / biography / causal / opinion
- Sinh 2 biến thể thêm từ template domain-specific → tổng 3 queries

---

## 4. Phương pháp Fine-tuning

### 4.1 Fine-tune Bi-Encoder

| Cài đặt | Giá trị |
|---|---|
| Base model | `all-MiniLM-L6-v2` |
| Loss | `MultipleNegativesRankingLoss` (in-batch negatives) |
| Batch | 8 × grad_accum=4 → effective batch 32 |
| Epochs | 3 |
| Max samples | 100,000 triplets |
| VRAM peak | ~2.5 GB |
| fp16 | ✓ |

**Dữ liệu huấn luyện:** triplets (query, positive_chunk, negative_chunk) được tạo từ BM25 hard negatives. Với mỗi query trong tập train, lấy positive từ qrels, negative là kết quả BM25 xếp hạng cao nhưng không liên quan.

### 4.2 Fine-tune Cross-Encoder

| Cài đặt | Giá trị |
|---|---|
| Base model | `ms-marco-MiniLM-L-6-v2` |
| Loss | Binary cross-entropy (label 1=relevant, 0=not) |
| Hard negatives | Top-20 BM25 results không có trong qrels |
| Batch | 4 × grad_accum=4 → effective batch 16 |
| Epochs | 3 |
| Max samples | 100,000 pairs |
| VRAM peak | ~2.0 GB |
| fp16 | ✓ |

---

## 5. Thiết kế Thực nghiệm

### 5.1 Dataset

| Thuộc tính | Giá trị |
|---|---|
| Dataset | BEIR trec-news-generated-queries |
| Corpus gốc | ~594K tài liệu tin tức |
| Sampling | 50% stratified by doc length, seed=42 |
| Chunks | ~1.56M (256 tokens, overlap 32) |
| Test queries | 216 queries (từ tập test split) |
| Qrels | BEIR format: {qid: {doc_id: relevance}} |

### 5.2 Ba cấu hình so sánh

| Cấu hình | Bi-Encoder | Cross-Encoder |
|---|---|---|
| **C1 — No Finetune** | `all-MiniLM-L6-v2` (base) | `ms-marco-MiniLM-L-6-v2` (base) |
| **C2 — Only BiE FT** | `models/bi_encoder` (finetuned) | `ms-marco-MiniLM-L-6-v2` (base) |
| **C3 — Full FT** | `models/bi_encoder` (finetuned) | `models/cross_encoder` (finetuned) |

### 5.3 Các giai đoạn đánh giá (6 stages)

Mỗi cấu hình được đánh giá trên 6 giai đoạn độc lập:

1. **BM25** — retrieval thưa (giống nhau 3 cấu hình, dùng làm baseline)
2. **Dense** — FAISS retrieval thuần túy
3. **Hybrid** — BM25 + Dense + RRF (1 query)
4. **HybridMultiQuery** — BM25 + Dense + RRF (3 query variants)
5. **MultiHybrid+BiEncoder** — stage 4 + bi-encoder rerank
6. **FullPipeline** — stage 5 + cross-encoder rerank

### 5.4 Metrics

- **NDCG@k** — chất lượng xếp hạng có trọng số
- **MAP@k** — mean average precision
- **Recall@k** — độ bao phủ tập relevant
- **MRR@10** — mean reciprocal rank (quan trọng cho QA)
- **Precision@k** — độ chính xác tập top-k

---

## 6. Kết quả

### 6.1 Bảng kết quả tổng hợp — NDCG@10

| Giai đoạn | C1 No FT | C2 Only BiE FT | C3 Full FT |
|---|---:|---:|---:|
| BM25 | 0.3689 | 0.3689 | 0.3689 |
| Dense | 0.2991 | **0.2893** | 0.2893 |
| Hybrid | 0.3901 | 0.3888 | 0.3888 |
| HybridMultiQuery | 0.3982 | 0.3932 | 0.3932 |
| MultiHybrid+BiEncoder | 0.3007 | **0.3613** | 0.3613 |
| **FullPipeline** | **0.4948** | **0.4880** | **0.3156** |

### 6.2 Bảng kết quả tổng hợp — MRR@10

| Giai đoạn | C1 No FT | C2 Only BiE FT | C3 Full FT |
|---|---:|---:|---:|
| BM25 | 0.3304 | 0.3304 | 0.3304 |
| Dense | 0.2560 | 0.2500 | 0.2500 |
| Hybrid | 0.3453 | 0.3436 | 0.3436 |
| HybridMultiQuery | 0.3497 | 0.3448 | 0.3448 |
| MultiHybrid+BiEncoder | 0.2575 | **0.3162** | 0.3162 |
| **FullPipeline** | **0.4496** | **0.4428** | **0.2545** |

### 6.3 Recall@10 và Recall@100

| Giai đoạn | C1 No FT | C2 Only BiE FT | C3 Full FT |
|---|---:|---:|---:|
| BM25 Recall@100 | 0.7132 | 0.7132 | 0.7132 |
| Dense Recall@100 | 0.6912 | 0.6703 | 0.6703 |
| Hybrid Recall@100 | 0.7832 | 0.7623 | 0.7623 |
| HybridMultiQuery Recall@10 | 0.5546 | 0.5505 | 0.5505 |
| MultiHybrid+BiEncoder Recall@10 | 0.4410 | 0.5081 | 0.5081 |
| **FullPipeline Recall@10** | **0.6426** | **0.6356** | **0.5176** |

---

## 7. Phân tích và Thảo luận

### 7.1 BM25 mạnh hơn Dense Retrieval trên domain tin tức

Dense retrieval (base) đạt NDCG@10 = **0.2991**, thấp hơn BM25 (**0.3689**) khoảng 19%. Điều này khá phổ biến với domain tin tức vì:
- Tin tức dùng từ ngữ cụ thể (tên người, địa điểm, sự kiện) — BM25 khớp chính xác từ khóa hiệu quả hơn
- `all-MiniLM-L6-v2` được huấn luyện trên dữ liệu tổng quát, chưa có semantic understanding tốt cho news

→ **Hybrid fusion là bắt buộc:** Kết hợp BM25 + Dense qua RRF tăng NDCG@10 lên **0.3901** (+5.7% so với BM25, +30% so với Dense).

### 7.2 Query Expansion đóng góp nhỏ nhưng nhất quán

Multi-query hybrid (3 variants) so với single-query hybrid:
- NDCG@10: 0.3888 → 0.3932 (+0.44%)
- Recall@10: 0.5390 → 0.5505 (+1.15%)
- MRR@10: 0.3436 → 0.3448 (+0.12%)

Gain khiêm tốn vì query expansion dùng rule-based template, không dùng T5 rewriter. Tuy nhiên chi phí thêm chỉ là 2× BM25 scoring + 2× FAISS search.

### 7.3 Fine-tune Bi-Encoder: cải thiện reranking nhưng giảm recall ở retrieval stage

Tại stage **MultiHybrid+BiEncoder**:
- C1 (base): NDCG@10 = 0.3007
- C2/C3 (finetuned): NDCG@10 = **0.3613** (+20.1%)

Fine-tuned bi-encoder cải thiện đáng kể khả năng **reranking** (đặt tài liệu liên quan lên đầu top-30). Tuy nhiên tại stage **Dense retrieval**:
- Base NDCG@10: 0.2991, Recall@100: 0.6912
- Finetuned NDCG@10: 0.2893, Recall@100: 0.6703

Mô hình fine-tuned có Recall@100 thấp hơn khi dùng cho FAISS retrieval. Điều này phản ánh trade-off: `MultipleNegativesRankingLoss` với in-batch negatives tối ưu cho ranking trong tập ứng viên, nhưng có thể giảm diversity trong không gian embedding.

### 7.4 Finding quan trọng: Fine-tune Cross-Encoder gây hại

Đây là kết quả đáng chú ý nhất:

| Config | FullPipeline NDCG@10 | FullPipeline MRR@10 |
|---|---:|---:|
| C1 — base CE | **0.4948** | **0.4496** |
| C2 — base CE | 0.4880 | 0.4428 |
| C3 — finetuned CE | 0.3156 | 0.2545 |

Fine-tuning cross-encoder làm NDCG@10 giảm từ 0.4948 xuống **0.3156** (−36.2%). Có thể giải thích bởi các nguyên nhân sau:

**a) Pretrained ms-marco CE đã transfer tốt sang news domain:**
MS MARCO chứa nhiều query dạng question-answering, bao gồm cả news-related queries. Model đã học cách đánh giá relevance tổng quát và không cần thêm fine-tuning.

**b) BM25 hard negatives không đại diện cho hard negatives thực tế của CE:**
Training CE với top-20 BM25 results làm negatives = dạy model phân biệt query với "documents khó theo BM25". Nhưng tại inference, CE nhận top-30 từ bi-encoder — phân phối hoàn toàn khác. Distribution shift giữa train và test gây degradation.

**c) Overfitting do dataset nhỏ:**
100K pairs với model có capacity lớn, chỉ 3 epochs — mô hình có thể memorize patterns của training set. BEIR evaluation đánh giá tổng quát hóa.

**d) Mất calibration của score:**
Base CE cho scores có ý nghĩa tuyệt đối (calibrated từ MS MARCO training). Fine-tuned CE thay đổi score distribution, có thể làm mất khả năng phân biệt relevant vs non-relevant.

> **Kết luận thực hành:** Với pretrained model chất lượng cao từ domain tương tự (MS MARCO → news), fine-tuning cross-encoder có thể **không cần thiết** và thậm chí **có hại**. Nên giữ base CE và tập trung fine-tune bi-encoder.

### 7.5 Tổng hợp giá trị của từng giai đoạn (C1 — best config)

```
BM25 baseline          NDCG@10 = 0.3689  (lexical matching)
+ Dense fusion         NDCG@10 = 0.3901  (+5.7%  — semantic coverage)
+ Query expansion      NDCG@10 = 0.3982  (+2.1%  — recall boost)
+ BiEncoder rerank     NDCG@10 = 0.3007  (↓ — stage reduction effect*)
+ CrossEncoder rerank  NDCG@10 = 0.4948  (+64%   — precision at top)
```

*Lưu ý: Stage BiEncoder rerank (top-30) cho NDCG@10 thấp hơn HybridMultiQuery vì kết quả bị thu hẹp — tuy nhiên đây là input tốt hơn cho CE.

---

## 8. Kết luận

### 8.1 Những điểm đã làm được

1. **Pipeline end-to-end** hoạt động trên 4 GB VRAM: từ retrieval đến generation
2. **Hybrid retrieval** nhất quán tốt hơn BM25 hoặc Dense đơn lẻ trên news corpus
3. **Fine-tuned Bi-Encoder** cải thiện rõ rệt giai đoạn reranking (+20% NDCG@10)
4. **Pretrained Cross-Encoder** (ms-marco) là thành phần đóng góp lớn nhất (+64% NDCG@10)

### 8.2 Những hạn chế và hướng cải thiện

| Hạn chế | Hướng cải thiện |
|---|---|
| Fine-tuned CE giảm hiệu suất | Thử mining hard negatives từ CE hoặc bi-encoder thay vì BM25 |
| Base Dense yếu hơn BM25 | Tăng max_train_samples, thử ANCE/DPR negative mining |
| Query expansion rule-based | Dùng T5/GPT để sinh query variants ngữ nghĩa đa dạng hơn |
| Chỉ đánh giá retrieval | Thêm evaluation end-to-end (answer correctness, faithfulness) |
| Bi-encoder fine-tune giảm recall | Thử contrastive learning với harder negatives, larger batch |

### 8.3 Bài học chính

> **Không phải lúc nào fine-tuning cũng có lợi.** Khi pretrained model đã được huấn luyện trên domain tương tự (MS MARCO → news), fine-tuning thêm với dữ liệu nhỏ và hard negatives không đại diện có thể gây overfitting và distribution shift tại inference. Đây là bài học quan trọng trong thực hành deep learning cho NLP/IR.

---

## 9. Phụ lục

### 9.1 Cài đặt phần cứng

| Thành phần | Cấu hình |
|---|---|
| GPU | NVIDIA RTX 3050 Ti — 4 GB VRAM |
| RAM | 16 GB DDR4 |
| Bộ nhớ | SSD |
| Python | 3.10 |
| PyTorch | 2.1+ với CUDA 12.1 |

### 9.2 Số liệu đầy đủ FullPipeline

| Metric | C1 No FT | C2 Only BiE | C3 Full FT |
|---|---:|---:|---:|
| NDCG@1 | 0.3542 | 0.3472 | 0.1551 |
| NDCG@3 | 0.4447 | 0.4400 | 0.2332 |
| NDCG@5 | 0.4736 | 0.4661 | 0.2719 |
| NDCG@10 | **0.4948** | 0.4880 | 0.3156 |
| MAP@10 | **0.4466** | 0.4398 | 0.2526 |
| Recall@5 | **0.5795** | 0.5702 | 0.3843 |
| Recall@10 | **0.6426** | 0.6356 | 0.5176 |
| MRR@10 | **0.4496** | 0.4428 | 0.2545 |

### 9.3 Files và Scripts

| File | Mục đích |
|---|---|
| `bi_encoder_training.py` | Fine-tune bi-encoder (MNRL) |
| `cross_encoder_training.py` | Fine-tune cross-encoder (BCE) |
| `evaluation.py` | Đánh giá 6 giai đoạn, 3 metrics groups |
| `eval_no_fintune` | Kết quả C1 |
| `eval_only_Bi_encode` | Kết quả C2 |
| `eval_full_pipeline` | Kết quả C3 |
| `retrieval_pipeline.py` | Pipeline inference |
| `rag_inference.py` | CLI entry point |
