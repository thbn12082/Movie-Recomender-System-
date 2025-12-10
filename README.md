# 🎬 MOVIE AI ULTIMATE - HỆ THỐNG GỢI Ý PHIM HYBRID

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![AI Model](https://img.shields.io/badge/Model-Hybrid%20(BERT%20%2B%20SVD)-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Họ và tên:** Nguyễn Thế Bình
> **Mã sinh viên:** [B22DCCN083]
> **Môn học:** Khoa học Dữ liệu (Data Science)

---

## 📌 I. TỔNG QUAN DỰ ÁN

Dự án xây dựng hệ thống gợi ý phim thông minh (Smart Movie Recommender), giải quyết bài toán quá tải thông tin cho người dùng. Hệ thống không chỉ gợi ý dựa trên nội dung phim mà còn phân tích hành vi cộng đồng và cảm xúc người dùng.

### 🌟 Điểm nổi bật:
1.  **Hiểu ngữ nghĩa (Semantic Understanding):** Sử dụng **BERT (Sentence-Transformers)** để "đọc hiểu" nội dung phim thay vì chỉ đếm từ (TF-IDF).
2.  **Mô hình lai (Hybrid Approach):** Kết hợp Content-Based (dựa trên nội dung) và Collaborative Filtering (dựa trên hành vi người dùng bằng SVD).
3.  **Gợi ý theo cảm xúc (Mood Filtering):** Bộ lọc phim theo tâm trạng (Vui, Buồn, Hồi hộp...).
4.  **Tương tác thời gian thực:** Lưu phim yêu thích và cập nhật gợi ý ngay lập tức.
5.  **Giao diện hiện đại:** Web app được xây dựng bằng Streamlit, dễ sử dụng và trực quan.

---

## 🛠 II. CÔNG NGHỆ & KỸ THUẬT

| Hạng mục | Công nghệ | Mục đích |
| :--- | :--- | :--- |
| **Ngôn ngữ** | Python | Xử lý backend và logic chính. |
| **Data** | Pandas, Numpy | Làm sạch và xử lý dữ liệu. |
| **NLP & AI** | **SBERT (all-MiniLM)** | Vector hóa văn bản phim (384 chiều). |
| **Model** | **TruncatedSVD** | Phân tích ma trận User-Item (Gợi ý cộng đồng). |
| **Frontend** | Streamlit | Giao diện người dùng. |

---

## 📂 III. CẤU TRÚC DỰ ÁN

Dự án được chuyển đổi từ Jupyter Notebook sang các file Python script (`.py`) để dễ dàng chạy và triển khai trên VS Code.

```text
KHDL_CK/
│
├── app.py                   # 🚀 CHÍNH: File chạy Web App (Streamlit)
├── 1_data_collection.py     # Thu thập dữ liệu từ TMDB API
├── 2_data_cleaning_eda.py   # Làm sạch dữ liệu và vẽ biểu đồ phân tích
├── 3_recommendation_models.py # Thử nghiệm và huấn luyện các mô hình gợi ý
├── 4_model_evaluation.py    # Đánh giá độ chính xác của mô hình
│
├── generate_bert.py         # Script tạo file vector BERT (chạy offline)
├── requirements.txt         # Các thư viện cần thiết
│
├── movies_final_dataset.csv # Dữ liệu phim đã xử lý
├── ratings.csv              # Dữ liệu đánh giá
├── bert_embeddings.npy      # File vector BERT (đã tính trước để chạy nhanh)
├── user_favorites.json      # Lưu lịch sử phim yêu thích của người dùng
│
└── venv/                    # Môi trường ảo Python (Virtual Environment)
```

---

## 🚀 IV. HƯỚNG DẪN CÀI ĐẶT & CHẠY (VS Code / Windows)

Để chạy dự án này trên máy tính của bạn, hãy làm theo các bước sau:

### Bước 1: Mở dự án
Mở thư mục `KHDL_CK` bằng **Visual Studio Code**.

### Bước 2: Tạo môi trường ảo (Virtual Environment)
Mở Terminal trong VS Code (`Ctrl + ~` hoặc menu `Terminal > New Terminal`) và chạy lệnh:

```powershell
python -m venv venv
```

### Bước 3: Kích hoạt môi trường và cài đặt thư viện
Kích hoạt môi trường ảo và cài các thư viện cần thiết từ file `requirements.txt`:

```powershell
# Kích hoạt môi trường ảo
.\venv\Scripts\Activate

# Cài đặt thư viện
pip install -r requirements.txt
```

### Bước 4: Chạy ứng dụng Web
Sau khi cài đặt xong, chạy lệnh sau để mở ứng dụng:

```powershell
streamlit run app.py
```
*(Hoặc dùng lệnh đầy đủ: `python -m streamlit run app.py`)*

👉 **Truy cập:** Trình duyệt sẽ tự động mở địa chỉ `http://localhost:8501`.

---

## 📊 V. QUY TRÌNH DỮ LIỆU (Data Pipeline)

Nếu bạn muốn chạy lại quy trình xử lý dữ liệu từ đầu, hãy chạy lần lượt các file sau:

1.  `python 1_data_collection.py`: Tải dữ liệu mới nhất từ TMDB (cần API Key).
2.  `python 2_data_cleaning_eda.py`: Làm sạch dữ liệu và sinh các biểu đồ thống kê.
3.  `python generate_bert.py`: Tạo lại file vector `bert_embeddings.npy` (Nặng, chạy sẽ lâu).

---

## 📜 VI. KẾT LUẬN

Project này minh họa trọn vẹn quy trình của một Data Scientist: Từ thu thập dữ liệu thô, làm sạch, xây dựng mô hình AI, đến triển khai sản phẩm thực tế.

*2025 - Nguyễn Thế Bình*