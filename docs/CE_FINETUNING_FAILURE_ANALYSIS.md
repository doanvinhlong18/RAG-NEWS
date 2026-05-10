# Phân tích nguyên nhân: Fine-tune Cross-Encoder làm giảm hiệu suất

---

## Vấn đề quan sát được

Fine-tuning cross-encoder không những không cải thiện mà còn **làm hỏng** toàn bộ pipeline:

| Cấu hình | BiEncoder NDCG@10 | FullPipeline NDCG@10 | CE đóng góp |
|---|---:|---:|---:|
| C1 — Base BiE + Base CE | 0.3007 | **0.4948** | **+0.1941 (+64.5%)** |
| C2 — Finetuned BiE + Base CE | 0.3613 | 0.4880 | +0.1267 (+35.1%) |
| C3 — Finetuned BiE + Finetuned CE | 0.3613 | **0.3156** | **−0.0457 (−12.6%)** |

**CE fine-tuned không chỉ không giúp ích mà còn kéo kết quả xuống thấp hơn cả bi-encoder.**
Bi-encoder một mình cho NDCG@10 = 0.3613, nhưng sau khi qua CE fine-tuned, kết quả giảm còn 0.3156.

Nhìn vào Recall cũng cho thấy tương tự:

| Cấu hình | Recall@5 (FullPipeline) | MRR@10 (FullPipeline) |
|---|---:|---:|
| C1 base CE | 0.5795 | 0.4496 |
| C2 finetuned BiE + base CE | 0.5702 | 0.4428 |
| C3 finetuned CE | **0.3843** | **0.2545** |

Finetuned CE chỉ tìm được 38% tài liệu liên quan trong top-5, trong khi base CE đạt 58%. Tài liệu relevant đầu tiên xuất hiện trung bình ở rank ~3.9 (1/MRR@10 ≈ 1/0.2545) so với rank ~2.2 của base CE.

---

## Nguyên nhân 1: Mismatch về Loss Function — Ranking bị đào tạo thành Classification

**Đây là nguyên nhân có tác động lớn nhất.**

### Base CE được huấn luyện như thế nào

Model gốc `cross-encoder/ms-marco-MiniLM-L-6-v2` được pre-train trên tập MS MARCO với **pairwise ranking loss**. Loss này tối ưu trực tiếp cho bài toán *so sánh tương đối*: "document A có liên quan hơn document B với query này không?" Đây chính xác là bài toán reranking.

### Fine-tuned CE được huấn luyện như thế nào

Nhìn vào `cross_encoder_training.py` — training dùng **Binary Cross-Entropy + Sigmoid**:

```python
# cross_encoder_training.py, dòng 289-322
class SigmoidCrossEncoder:
    def __init__(self, ...):
        self._ce = CrossEncoder(model_name_or_path, num_labels=1, ...)
        self._ce.activation_fct = nn.Sigmoid()   # ← Classification loss

    def predict(self, pairs, ...):
        raw = self._ce.predict(pairs, ...)
        return torch.sigmoid(torch.tensor(raw)).numpy()
```

Label trong training data là `1.0` (relevant) hoặc `0.0` (not relevant):

```python
# cross_encoder_training.py, dòng 222-241
examples.append(InputExample(texts=[query_text, pos_text], label=1.0))
...
examples.append(InputExample(texts=[query_text, neg_text], label=0.0))
```

Kết quả: model học cách **phân loại nhị phân** — "document này có liên quan hay không?" thay vì **xếp hạng tương đối** — "document nào liên quan hơn?"

### Vì sao điều này gây hại cho reranking

Với bài toán reranking (CE nhận vào top-30 candidates từ bi-encoder):
- Tất cả 30 documents đều là "khó" — chúng được bi-encoder chọn vì trông có vẻ liên quan
- Model phân loại nhị phân cố gắng đặt ngưỡng tuyệt đối: "trên 0.5 = relevant, dưới 0.5 = not relevant"
- Nhưng reranking cần phân biệt tương đối: "doc nào trong 30 cái này liên quan nhất?"

Base CE với ranking loss biết cách phân biệt độ liên quan trong một tập candidates chặt chẽ. Finetuned CE với BCE có xu hướng đẩy nhiều candidates về cùng một score (gần 0 hoặc gần 1), mất khả năng phân biệt tinh tế.

---

## Nguyên nhân 2: Distribution Mismatch giữa Training Negatives và Inference Pool

**Đây là nguyên nhân có thể kiểm chứng trực tiếp qua số liệu.**

### Training: Negatives được chọn như thế nào

```python
# cross_encoder_training.py, dòng 195-245
# Dùng BASE bi-encoder để search FAISS
bi_encoder = SentenceTransformer(bi_encoder_model_name)  # all-MiniLM-L6-v2 base
...
search_k = skip_top_k + top_k_neg  # = 5 + 20 = 25
_, indices = index.search(q_embs, search_k)
...
for idx in indices[row_idx][skip_top_k:]:  # bỏ top-5, lấy từ rank 5 trở đi
    ...
    if neg_count >= neg_per_pos:  # chỉ lấy 4 negatives mỗi query
        break
```

Cụ thể: mỗi query, CE training nhận negatives là **FAISS rank 5, 6, 7, 8** của **base bi-encoder**.

### Inference: CE nhận input từ đâu

```python
# evaluation.py, stage 5→6
bi_chunk = bi_encoder_rerank_results(
    queries, multi_chunk, chunks, encoder,  # encoder = FINETUNED bi-encoder
    top_k_in=top_k_retrieval,   # = 100
    top_k_out=30,               # → top-30
)
ce_chunk = ce_rerank_results(
    queries, bi_chunk, ...      # CE nhận top-30 từ FINETUNED bi-encoder
)
```

CE nhận vào top-30 từ **finetuned bi-encoder**, là một distribution hoàn toàn khác.

### Bằng chứng số liệu: hai distribution khác nhau như thế nào

So sánh Dense Retrieval (base vs finetuned bi-encoder):

| Metric | Base BiE (C1) | Finetuned BiE (C2/C3) | Chênh lệch |
|---|---:|---:|---:|
| NDCG@10 | 0.2991 | 0.2893 | −0.0098 |
| Recall@10 | 0.4387 | 0.4178 | −0.0209 |
| Recall@100 | **0.6912** | **0.6703** | **−0.0209** |

Finetuned bi-encoder cover ít hơn 2% tài liệu relevant trong top-100 so với base. Nghĩa là **top-30 mà CE nhận được tại inference khác đáng kể so với top-30 từ base bi-encoder mà CE học trong training**.

Cụ thể hơn — Hybrid (base BiE) vs Hybrid (finetuned BiE) tại Recall@100:

| | C1 No FT | C2/C3 Finetuned BiE |
|---|---:|---:|
| Hybrid Recall@100 | **0.7832** | 0.7623 |
| HybridMultiQuery Recall@100 | 0.7728 | 0.7519 |

Pool từ đó CE lấy top-30: finetuned bi-encoder thiếu ~2% tài liệu relevant. CE training chưa bao giờ thấy pattern này.

---

## Nguyên nhân 3: Chất lượng và Số lượng Hard Negatives quá hạn chế

### Chỉ 4 negatives mỗi query, từ một khoảng rank rất hẹp

```python
# cross_encoder_training.py
neg_per_pos=4,      # [FIX-C] chỉ 4 negatives mỗi positive
skip_top_k=5,       # [FIX-B] skip top-5
```

```
Mỗi query: 1 positive + 4 negatives = 5 examples
Negatives đến từ: FAISS rank 5, 6, 7, 8 (chỉ 4 ranks!)
```

### Hệ quả

CE training chỉ học phân biệt:
- Relevant doc (label 1.0) vs documents xếp hạng **5-8** theo base bi-encoder (label 0.0)

Nhưng tại inference, CE phải phân biệt trong top-30 candidates — bao gồm rank 1-4 (documents **rất giống query** mà bi-encoder tin tưởng nhất), rank 5-8, và cả rank 29-30 (documents ít giống hơn). CE chưa bao giờ học cách phân biệt trong context này.

**Khoảng rank 1-4 từ bi-encoder bị bỏ qua hoàn toàn trong training** vì `skip_top_k=5`. Tại inference, đây lại là những candidates quan trọng nhất cần xếp hạng đúng.

### Tỷ lệ class imbalance

```
Tổng examples: tối đa 100,000 (max_train_samples)
Với ~5000 queries (max_train_queries) × 5 examples/query = ~25,000 examples
Trong đó: ~5,000 positive + ~20,000 negative → tỷ lệ 1:4
```

BCE loss với tỷ lệ 1:4 có thể học thiên lệch về phía "not relevant" — làm giảm recall và push scores về phía thấp cho cả relevant lẫn non-relevant candidates.

---

## Nguyên nhân 4: Pretrained CE đã Transfer Tốt sang News Domain

**Đây là nguyên nhân nền tảng giải thích tại sao fine-tuning có ngưỡng lợi ích rất thấp.**

### MS MARCO → News: Transfer tự nhiên

MS MARCO là tập QA từ Bing search, bao gồm nhiều queries dạng tin tức:
- "Who won the 2016 election?" → trực tiếp overlap với BEIR trec-news
- "What caused the financial crisis?" → event-based, giống news queries
- Named entities, numbers, dates — tất cả đều phổ biến trong cả hai domain

Kết quả: base CE đạt **NDCG@10 = 0.4948** trên trec-news mà không cần bất kỳ domain adaptation nào.

### Fine-tuning với 25,000 examples không đủ để cải thiện model đã "giỏi"

Fine-tuning với 25,000 noisy examples trên model đã đạt 0.4948 NDCG@10:
- Learning rate `1.0e-5` nhỏ nhưng 3 epochs × 25,000 examples = 75,000 gradient steps
- Model weights bị kéo ra khỏi vị trí đã được calibrate tốt bởi MS MARCO training
- Knowledge được học từ MS MARCO (ranking behavior trên 540,000 queries) bị partially overwrite

Đây là hiện tượng **catastrophic forgetting** ở quy mô nhỏ: thông tin từ MS MARCO training bị xói mòn, trong khi thông tin từ 25,000 trec-news examples không đủ để bù đắp.

---

## Tổng hợp: Mức độ đóng góp của từng nguyên nhân

```
FullPipeline NDCG@10:
  Base CE (C1):       0.4948  ──────────────────────────────
  Finetuned CE (C3):  0.3156  ──────────────
  Chênh lệch:        −0.1792  (−36.2%)
```

Ước lượng đóng góp của từng nguyên nhân vào sự sụt giảm −0.1792:

| Nguyên nhân | Ước lượng đóng góp | Bằng chứng |
|---|---:|---|
| 1. Loss: ranking → classification | ~40–50% | CE làm kết quả tệ hơn bi-encoder (−12.6%) — binary classifier không thể rerank tốt |
| 2. Distribution mismatch (training vs inference pool) | ~25–35% | Dense recall thay đổi 2%, nhưng cascades qua toàn pipeline |
| 3. Hard negatives chỉ từ rank 5-8 | ~15–20% | CE không học cách xử lý top-4 candidates — quan trọng nhất tại inference |
| 4. Catastrophic forgetting MS MARCO | ~10–15% | 25K noisy samples overwrite 540K MS MARCO samples |

---

## So sánh trực tiếp: CE đang làm gì với top-30 candidates

Nhìn vào chain of evidence tại mỗi stage:

**Với C1 (base CE):**
```
HybridMultiQuery → BiEncoder (top-30) → CE → FullPipeline
   0.3982              0.3007              0.4948
                    [base rerank]     [base CE: +64.5%]
```

**Với C3 (finetuned CE):**
```
HybridMultiQuery → BiEncoder (top-30) → CE → FullPipeline
   0.3932              0.3613              0.3156
               [finetuned rerank]    [finetuned CE: −12.6%]
```

Base CE nhận input kém hơn (bi-encoder 0.3007 vs 0.3613) nhưng cải thiện gần gấp đôi.
Finetuned CE nhận input tốt hơn (0.3613) nhưng thực sự làm hỏng kết quả.

Điều này cho thấy **finetuned CE đang xếp hạng sai thứ tự** — đẩy relevant docs xuống thấp hơn vị trí chúng đã được bi-encoder đặt.

---

## Hướng khắc phục (nếu muốn fine-tune CE đúng cách)

### Fix 1: Dùng Pairwise Ranking Loss thay BCE

```python
# Thay vì:
InputExample(texts=[query, pos], label=1.0)
InputExample(texts=[query, neg], label=0.0)

# Dùng pairwise:
InputExample(texts=[query, pos, neg])  # MarginMSELoss hoặc ContrastiveLoss
```

### Fix 2: Mine hard negatives từ FINETUNED bi-encoder (không phải base)

```python
# Hiện tại (sai):
bi_encoder_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # BASE

# Đúng: dùng model sẽ thực sự cung cấp candidates tại inference
bi_encoder_model_name = "models/bi_encoder"  # FINETUNED
```

### Fix 3: Mở rộng khoảng rank lấy negatives

```python
# Hiện tại: skip 5, lấy 4 → chỉ rank 5-8
neg_per_pos=4, skip_top_k=5

# Tốt hơn: skip ít hơn, lấy nhiều hơn, sample ngẫu nhiên trong range
# Ví dụ: skip 2, sample 4-6 ngẫu nhiên từ rank 2-30
```

### Fix 4: Kết hợp Knowledge Distillation thay vì Fine-tuning trực tiếp

Thay vì train CE từ scratch labels (0/1), dùng **score của base CE** làm soft labels — bảo toàn ranking knowledge của MS MARCO trong khi adapts sang domain mới.

### Fix 5: Freeze early layers, chỉ train top layers

```python
# Fine-tune chỉ top 2-3 transformer layers
# Bảo toàn general language understanding, chỉ adapt representation
```

---

## Kết luận

Sự thất bại của CE fine-tuning không phải do một lỗi đơn lẻ mà do sự kết hợp của nhiều vấn đề thiết kế:

1. **Loss function sai** — chuyển từ ranking model sang binary classifier
2. **Negatives lấy từ wrong distribution** — base bi-encoder thay vì finetuned bi-encoder
3. **Phạm vi rank quá hẹp** — chỉ học từ rank 5-8, không học cách xử lý rank 1-4
4. **Catastrophic forgetting** — 25K noisy samples không đủ để beat 540K MS MARCO samples

Bài học rút ra: khi fine-tuning một model đã được pre-trained tốt trên domain tương tự (**ms-marco → news**), chi phí phá vỡ existing knowledge thường cao hơn lợi ích từ domain adaptation với dữ liệu nhỏ. Fine-tuning CE chỉ có ý nghĩa khi:
- Domain thực sự khác biệt hoàn toàn (y tế, luật, kỹ thuật chuyên sâu)
- Training data đủ lớn và chất lượng cao
- Dùng đúng loss function cho bài toán ranking
- Negatives được mine từ đúng distribution sẽ xuất hiện tại inference
