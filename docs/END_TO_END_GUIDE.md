# Hướng Dẫn Chạy Toàn Bộ Dự Án (End-to-End Guide)

Tài liệu này hướng dẫn chi tiết quy trình chạy toàn bộ hệ thống RAG (Retrieval-Augmented Generation) cho News Intelligence từ bước xử lý dữ liệu thô ban đầu cho tới bước sinh câu trả lời tự động.

## 1. Cài đặt Môi Trường (Environment Setup)

Trước khi bắt đầu, đảm bảo rằng bạn đã kích hoạt môi trường ảo (virtual environment) và cài đặt đầy đủ các thư viện cần thiết.

```bash
# Cài đặt các thư viện từ requirements.txt
pip install -r requirements.txt
```

*(Lưu ý: Nếu sử dụng GPU, đảm bảo đã cài đặt đúng phiên bản PyTorch hỗ trợ CUDA phù hợp với máy của bạn).*

---

## 2. Tiền Xử Lý và Phân Tích Dữ Liệu (Data Analysis & Preprocessing)

Bước đầu tiên là chuẩn bị tập dữ liệu, làm sạch, chia đoạn (chunking), lọc bỏ các dữ liệu rác/trùng lặp và tạo các biểu đồ phân tích.

**Script thực thi:** `data_analysis_preprocessing.py`

```bash
python data_analysis_preprocessing.py --chunk_size 256 --overlap 32 --remove_stopwords
```
* **Ý nghĩa:** Chạy pipeline để lấy mẫu dữ liệu (xem chi tiết tại [docs/sampling_methodology.md](sampling_methodology.md)), phân tích EDA, lọc outlier, bỏ trùng lặp và tiến hành chunking với giới hạn 256 token/chunk.
* **Đầu ra (Outputs):**
  * Thư mục `outputs/before_preprocessing/`: Biểu đồ trước xử lý.
  * Thư mục `outputs/after_preprocessing/`: Biểu đồ sau xử lý.
  * Thư mục `outputs/data/sampled/`: Dữ liệu đã được lấy mẫu (`sampled_corpus.jsonl`, `sampled_queries.json`, `sampled_qrels.tsv`).
  * Thư mục `outputs/data/cleaned/`: Dữ liệu gốc đã được làm sạch (`corpus.jsonl`, `queries.json`, `qrels.tsv`).
  * Thư mục `outputs/data/processed/`: Dữ liệu phân mảnh (`chunks.jsonl`, `tokenized_corpus.pkl`) cùng với log `processing_summary.txt`.

---

## 3. Huấn Luyện Mô Hình (Model Fine-Tuning) - *[Tùy chọn]*

Nếu bạn muốn tùy chỉnh (fine-tune) mô hình embedding cho phù hợp hơn với miền dữ liệu bài báo (News Intelligence):

### 3.1. Huấn luyện Bi-Encoder
Dùng để tạo ra các Vector Embeddings tối ưu cho tài liệu.
**Script thực thi:** `bi_encoder_training.py`
```bash
python bi_encoder_training.py
```

### 3.2. Huấn luyện Cross-Encoder
Dùng để Rerank (chấm điểm lại) các kết quả sau khi đã truy xuất bằng Bi-Encoder.
**Script thực thi:** `cross_encoder_training.py`
```bash
python cross_encoder_training.py
```

*(Lưu ý: Các thiết lập về batch size, learning rate, epoch có thể điều chỉnh trong `config.yaml` hoặc trực tiếp trong file mã nguồn. Bước này tốn nhiều tài nguyên GPU).*

---

## 4. Trích Xuất Đặc Trưng và Đánh Chỉ Mục (Indexing)

Sau khi có dữ liệu sạch và mô hình nhúng (Embedding), chúng ta cần vector hóa toàn bộ dữ liệu và xây dựng các hệ thống tra cứu (BM25 cho từ khóa và FAISS cho ngữ nghĩa).

**Script thực thi:** `indexing.py`

```bash
python indexing.py
```
* **Ý nghĩa:** Đọc các chunks từ `outputs/data/processed/chunks.jsonl`. Tiến hành nhúng (embed) bằng mô hình Bi-Encoder và lưu thành FAISS Index. Đồng thời tạo BM25 Index cho tìm kiếm từ khóa.
* **Đầu ra:** Các file `.index`, `.pkl` lưu trữ trong thư mục được chỉ định ở `config.yaml`.

---

## 5. Truy Xuất và Đánh Giá (Retrieval & Evaluation)

Tiến hành truy xuất dữ liệu lai (Hybrid Search: BM25 + FAISS Vector) và đánh giá độ chính xác của hệ thống bằng các độ đo tiêu chuẩn của Information Retrieval (như NDCG, MRR, Recall).

**Script thực thi:** `evaluation.py`

```bash
python evaluation.py
```
* **Ý nghĩa:** Chạy toàn bộ các queries có trong tập test (từ `qrels`), kết hợp giữa thuật toán tìm kiếm truyền thống và AI, áp dụng thuật toán hợp nhất RRF (Reciprocal Rank Fusion) và dùng Cross-Encoder để rerank kết quả. In ra báo cáo hiệu năng chi tiết.

---

## 6. Suy Luận Đầu Cuối (End-to-End RAG Inference)

Chạy pipeline tổng thể thực tế: Nhập một câu hỏi từ người dùng -> Truy xuất thông tin (Retrieval) -> Đưa vào LLM Sinh văn bản (Generation).

**Script thực thi:** `rag_inference.py`

```bash
python rag_inference.py
```
* **Ý nghĩa:** Khởi tạo hệ thống Chatbot/RAG. Bạn có thể truyền vào một câu query bất kỳ. Hệ thống sẽ:
  1. Trích xuất (Retrieve) top các bài báo tin tức liên quan nhất.
  2. Đưa context bài báo vào mô hình Ngôn Ngữ Lớn (LLM - như Mistral, T5...).
  3. LLM trả về câu trả lời tổng hợp cùng trích dẫn dựa trên nội dung tìm được.

---

## Tóm Lược Luồng Chạy (Flow Summary)
1. **Config** -> Điều chỉnh cấu hình trong `config.yaml`
2. **Data** -> `python data_analysis_preprocessing.py`
3. **Train** -> `python bi_encoder_training.py` (tùy chọn)
4. **Index** -> `python indexing.py`
5. **Eval** -> `python evaluation.py`
6. **Chat** -> `python rag_inference.py`
