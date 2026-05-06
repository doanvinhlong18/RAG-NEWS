# So Sánh SparseBM25 và rank_bm25 (BM25Okapi)

Tài liệu này giải thích lý do tại sao hệ thống RAG-NEWS không sử dụng thư viện phổ biến `rank_bm25` cho module truy xuất từ khóa, mà thay vào đó sử dụng một thuật toán tự viết lại mang tên `SparseBM25`.

## 1. Vấn Đề Gặp Phải: Lỗi Tràn Bộ Nhớ (OOM - Out Of Memory)

Trong quá trình xây dựng Inverted Index cho BM25 với các tập dữ liệu thực tế (như BeIR/trec-news với hơn hàng triệu bài báo), hệ thống liên tục gặp lỗi sập bộ nhớ (MemoryError / OOM).

Sau khi kiểm tra, chúng tôi phát hiện 2 điểm nghẽn trí mạng từ `rank_bm25`:
1. **Lúc nạp dữ liệu:** Nó yêu cầu phải đưa toàn bộ mảng token của toàn bộ corpus (danh sách chứa danh sách các chuỗi) vào bộ nhớ cùng lúc.
2. **Lúc lưu trữ (Serialization):** Đây là nguyên nhân chính. Cấu trúc dữ liệu nội bộ của `rank_bm25.BM25Okapi` lưu tần suất từ vựng (Term Frequencies) dưới dạng một **Python List chứa các Dictionary** (mỗi bài báo là 1 dictionary). Việc lưu trữ 1.5 triệu Dictionaries trên RAM gây ra Overhead cực kỳ khổng lồ. Khi gọi hàm `pickle.dump()`, Python tạo ra các bộ đệm (memo) khổng lồ để đóng gói hàng triệu object này, dẫn tới lỗi `MemoryError` sập RAM lập tức, ngay cả trên máy tính có RAM dung lượng vừa phải.

## 2. Giải Pháp: SparseBM25

Để giải quyết, chúng tôi tự viết lại thuật toán trong file `data_pipeline/sparse_bm25.py`.

`SparseBM25` tận dụng module `CountVectorizer` của `scikit-learn` và cấu trúc ma trận thưa (Compressed Sparse Column - CSC) của `scipy`. Những thư viện này được viết bằng C++ và tối ưu hóa cực sâu cho việc tính toán đại số tuyến tính trên không gian cực lớn chứa nhiều số 0.

### So Sánh Nhanh

| Tiêu Chí | `rank_bm25.BM25Okapi` | `SparseBM25` (Tự viết) |
| :--- | :--- | :--- |
| **Cấu trúc dữ liệu** | `list` of `dict` (Thuần Python) | `scipy.sparse.csc_matrix` (C++) |
| **Overhead bộ nhớ** | Rất lớn (~hàng GB cho 1M docs) | Siêu nhỏ (~vài chục MB cho 1M docs) |
| **Tốc độ truy vấn** | Chậm (Quét hàm `.get()` trên List) | Cực nhanh (Slice cột ma trận C++) |
| **Serialization (Pickle)** | Thường xuyên gây OOM với data lớn | Siêu nhẹ, siêu nhanh |
| **Input Token** | Yêu cầu List of Lists đưa vào RAM | Hỗ trợ Python Generator (Streaming) |

## 3. Tính Toàn Vẹn Toán Học (Mathematical Equivalence)

Mặc dù thay đổi hoàn toàn về kiến trúc bộ nhớ, `SparseBM25` được thiết kế để **đảm bảo kết quả đầu ra (BM25 Score) giống hệt 100%** so với `rank_bm25.BM25Okapi`. 

Chúng tôi tái tạo lại chính xác các công thức từ thư viện gốc:

- **Tính IDF (Inverse Document Frequency):**
  - Công thức gốc: `math.log(corpus_size - nd + 0.5) - math.log(nd + 0.5)`
  - Được ánh xạ trực tiếp sang Numpy Matrix.
  - Xử lý các IDF âm (Negative IDFs): Giữ nguyên logic biến đổi các từ quá phổ biến (có IDF âm) thành các giá trị dương cực nhỏ dựa trên trung bình epsilon.

- **Tính Điểm BM25:**
  - Áp dụng công thức Okapi: `IDF * (TF * (k1 + 1)) / (TF + k1 * (1 - b + b * (doc_len / avgdl)))`
  - Các tham số cốt lõi `k1=1.5` và `b=0.75` được giữ nguyên.
  - Điểm khác biệt duy nhất là: thay vì cộng dồn điểm bằng vòng lặp `for` với các phần tử `0`, `SparseBM25` lọc ra các document có chứa từ khóa đó (`q_tf > 0`) và áp dụng tính toán vector hóa (Vectorized computation) của Numpy, giúp loại bỏ các phép tính thừa.

## 4. Tổng Kết

Bằng việc tự triển khai `SparseBM25`, pipeline không phải phụ thuộc vào thư viện bên thứ 3 chậm chạp, hoàn toàn vượt qua rào cản phần cứng khi xử lý Big Data, cho phép hệ thống RAG-NEWS chạy nội bộ ngay cả trên phần cứng cá nhân bị giới hạn bộ nhớ (ví dụ: 4GB - 8GB RAM).
