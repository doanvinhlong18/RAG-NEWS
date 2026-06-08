# Phương pháp Lấy mẫu Phân tầng (Stratified Sampling Methodology)

Tài liệu này ghi chép lại chi tiết phương pháp và thuật toán được sử dụng để giảm thiểu kích thước bộ dữ liệu (dataset) xuống còn 50% trong 파ipeline RAG, đồng thời vẫn bảo toàn được phân phối gốc của dữ liệu, tránh hiện tượng lệch nhãn (label skew) và lệch đặc trưng (feature skew).

## 1. Mục tiêu (Objectives)
- **Giảm dung lượng:** Thu gọn dataset (bao gồm Corpus, Queries, và Qrels) xuống còn khoảng 50% so với ban đầu nhằm tăng tốc độ tiền xử lý, embedding và huấn luyện.
- **Bảo toàn phân phối:** Không sử dụng random sampling thông thường để tránh việc các nhóm dữ liệu thiểu số bị loại bỏ hoàn toàn.
- **Tính toàn vẹn (Integrity):** Đảm bảo không làm mất liên kết giữa Câu hỏi (Query) và Tài liệu liên quan (Relevant Document) đã được gán nhãn trong `qrels`.
- **Hiệu năng:** Tối ưu hóa bộ nhớ RAM, chạy nhanh gọn bằng cách chỉ thao tác với Metadata (ID, độ dài).

## 2. Tiêu chí Phân tầng (Stratification Criteria)

Do BEIR dataset không có các "nhãn phân loại" rõ ràng (như chủ đề bài báo), chúng ta tiến hành tạo các nhóm (bins/strata) dựa trên đặc trưng thống kê của chính dữ liệu:

### 2.1 Đối với Queries
Mỗi Query được đánh giá dựa trên 2 tiêu chí:
1. **Số lượng Qrels (`num_qrels`):** Số lượng tài liệu liên quan đến câu hỏi đó (VD: 1, 2-5, 6+).
2. **Độ dài trung bình của tài liệu liên quan (`avg_doc_len`):** Đo bằng số lượng từ (words) trung bình của các documents được gán nhãn cho query đó. 

### 2.2 Đối với Corpus (Documents)
Mỗi Document được đánh giá dựa trên:
1. **Độ dài văn bản (Doc Length):** Lấy tổng số từ. Hệ thống sẽ tự động tính toán các phân vị (Quantiles - chia làm 3 khoảng: Ngắn, Trung bình, Dài) dựa trên phân phối thực tế của toàn bộ Corpus để làm mốc phân tầng.

## 3. Quy trình Thực hiện (Workflow)

Pipeline lấy mẫu (được triển khai tại `sampling_utils.py`) trải qua 5 bước chính:

### Bước 1: Khởi tạo và Phân tích Phân phối Gốc
- Lấy Seed cố định (mặc định = 42) để đảm bảo tính tái lập (reproducibility).
- Quét qua toàn bộ Corpus để tính toán chiều dài của từng Document (lưu vào dictionary để tối ưu hiệu năng).
- Tính toán 2 mốc phân vị (Tertiles - 33% và 66%) của độ dài văn bản để chia toàn bộ Corpus gốc thành 3 Bins: Ngắn, Trung bình, Dài. Thống kê số lượng Document trong từng Bin này để làm mục tiêu (Target Distribution) cho quá trình lấy mẫu bù sau này.

### Bước 2: Phân tầng Queries (Query Stratification)
- Với mỗi Query, tính toán `num_qrels` và `avg_doc_len`.
- Tính toán các mốc phân vị (Quantiles) cho cả 2 chỉ số này trên toàn bộ tập Queries.
- Nhóm các Queries thành các "Strata" (Phân tầng) dựa trên tổ hợp của 2 chỉ số trên (VD: Query có ít Qrels + Độ dài Doc ngắn sẽ vào chung một nhóm).

### Bước 3: Lấy mẫu Queries (Sampling Queries)
- Tại mỗi Stratum (nhóm Queries đã phân tầng), sử dụng ngẫu nhiên `random.sample()` để chọn ra đúng **50%** (theo `ratio`) số lượng Queries.
- Cách làm này đảm bảo rằng dù là loại Query phổ biến hay loại Query hiếm gặp (có rất nhiều docs liên quan, hoặc docs cực dài) đều sẽ được giữ lại theo đúng tỷ lệ 50%.

### Bước 4: Khôi phục Toàn vẹn Nhãn (Integrity Check)
- Sau khi có danh sách 50% Queries, hệ thống tiến hành duyệt lại `qrels`.
- **Quy tắc cốt lõi:** Bắt buộc phải giữ lại 100% các Documents có xuất hiện trong `qrels` của những Queries đã được chọn. 
- Điều này giúp các nhãn relevance không bị suy giảm hay mất mát.

### Bước 5: Bù đắp Corpus theo Phân phối (Corpus Padding)
- Tính toán lượng Documents còn thiếu để đạt được mục tiêu 50% tổng dung lượng Corpus ban đầu.
- So sánh số lượng Documents hiện đã chọn (từ Bước 4) với Mục tiêu Phân phối (Target Distribution ở Bước 1) trong từng nhóm chiều dài (Ngắn, Trung bình, Dài).
- Lấy mẫu ngẫu nhiên phần Documents **không liên quan** (Unrelated Docs) thuộc từng Bin chiều dài để bù đắp đúng vào số lượng còn thiếu của Bin đó.
- *Kết quả:* Tổng số lượng Document đạt đúng 50%, và phân phối độ dài (chuông Gauss/Lệch) của dataset sau lấy mẫu khớp hoàn toàn với dataset gốc.

## 4. Ưu điểm của Phương pháp (Advantages)

1. **Memory Efficient (Tiết kiệm RAM):** Thuật toán chỉ làm việc trên các biến `ID` và `độ dài văn bản` (số nguyên). Nó không sao chép chuỗi ký tự (strings) của hàng triệu Documents ra thành nhiều phiên bản, do đó tránh được lỗi Out of Memory (OOM).
2. **Chống Data Leakage:** Việc lấy mẫu bù đắp (Padding) chỉ sử dụng các Documents "không liên quan", đảm bảo không sinh ra các nhãn giả hoặc làm mất các nhãn thật.
3. **Độ tích hợp cao:** Dữ liệu sau khi chọn lọc ID sẽ được đóng gói lại thành các Dictionaries chuẩn xác cấu trúc BEIR để truyền vào hệ thống Chunking & Tokenization của Pipeline bên dưới mà không làm thay đổi các mã nguồn inference.
