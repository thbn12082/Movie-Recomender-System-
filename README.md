# 🎬 MOVIE AI ULTIMATE - HYBRID RECOMMENDATION SYSTEM

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![AI Model](https://img.shields.io/badge/Model-Hybrid%20(BERT%20%2B%20SVD)-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **Họ và tên:** Nguyễn Thế Bình
> **Mã sinh viên:** [B22DCCN083]
> **Môn học:** Khoa học Dữ liệu (Data Science)
> **Link Demo:** [DÁN LINK STREAMLIT APP CỦA BẠN VÀO ĐÂY]

---

## 📌 I. TỔNG QUAN DỰ ÁN

Dự án xây dựng hệ thống gợi ý phim thông minh (Smart Movie Recommender), giải quyết bài toán quá tải thông tin cho người dùng. Hệ thống không chỉ gợi ý dựa trên kịch bản phim mà còn thấu hiểu hành vi cộng đồng và ngữ cảnh cảm xúc của người dùng.

### 🌟 Điểm nổi bật (Advanced Features):
Dự án đã hoàn thành **100% yêu cầu cơ bản** và **5/5 yêu cầu nâng cao**:
1.  **Advanced Embeddings:** Sử dụng **BERT (Sentence-Transformers)** để hiểu ngữ nghĩa nội dung phim (thay vì chỉ dùng TF-IDF).
2.  **Hybrid Approach:** Kết hợp Content-Based và Collaborative Filtering (SVD).
3.  **Real-time Recommendation:** Gợi ý thay đổi tức thì ngay khi người dùng tương tác (Like/Add to History).
4.  **Context-Aware:** Bộ lọc theo **Tâm trạng (Mood Filtering)** - Gợi ý phim phù hợp với cảm xúc (Vui, Buồn, Hồi hộp...).
5.  **User Session:** Lưu lịch sử người dùng trong phiên làm việc.

---

## 🛠 II. CÔNG NGHỆ & KỸ THUẬT

| Hạng mục | Công nghệ sử dụng | Mục đích |
| :--- | :--- | :--- |
| **Ngôn ngữ** | Python | Xử lý backend và logic chính. |
| **Data Cleaning** | Pandas, Numpy | Xử lý Missing values, Duplicate, Outlier, Normalization. |
| **NLP & AI** | **SBERT (all-MiniLM-L6-v2)** | Vector hóa văn bản nâng cao (Deep Learning). |
| **Matrix Factorization** | **TruncatedSVD** (Sklearn) | Phân tích hành vi người dùng (Collaborative Filtering). |
| **API** | TMDB API | Làm giàu dữ liệu (Lấy Poster, Overview chuẩn). |
| **Frontend** | Streamlit | Giao diện Web tương tác, Responsive. |
| **Deployment** | Streamlit Cloud | Triển khai ứng dụng lên máy chủ đám mây. |

---

## 📊 III. QUY TRÌNH THỰC HIỆN (PIPELINE)

### 1. Thu thập & Làm sạch dữ liệu
* **Nguồn:** MovieLens Small + TMDB API.
* **Quy mô:** ~9,700 bộ phim và 100,000 lượt đánh giá.
* **Kỹ thuật làm sạch (5/5 tác vụ):**
    * [x] Xử lý Missing Values (Overview).
    * [x] Loại bỏ Duplicate phim.
    * [x] Xử lý Outlier (Lọc bỏ phim < 10 votes).
    * [x] Chuẩn hóa dữ liệu Rating (Min-Max Scaling).
    * [x] Vector hóa văn bản (TF-IDF & BERT).

### 2. Xây dựng Mô hình (Hybrid Model)

#### A. Content-Based Filtering (Nâng cấp với BERT)
* **Vấn đề:** TF-IDF cũ không hiểu ngữ nghĩa (VD: "Space" và "Galaxy" được coi là khác nhau).
* **Giải pháp:** Sử dụng mô hình **BERT (all-MiniLM-L6-v2)** để biến tóm tắt phim thành vector 384 chiều.
* **Kỹ thuật:** *Offline Inference* (Tính toán trước vector và lưu vào file `.npy` để tăng tốc độ Web khi deploy).

#### B. Collaborative Filtering (User-Based)
* **Thuật toán:** Matrix Factorization sử dụng **SVD (Singular Value Decomposition)**.
* **Cơ chế:** Nén ma trận User-Item khổng lồ xuống 20 chiều không gian tiềm ẩn (Latent Features) để tìm ra "gu" ngầm của người dùng.
* **Đánh giá:**
    * RMSE (Root Mean Squared Error): **~0.85** (Đã tối ưu bằng phương pháp Mean Centering).
    * MAE (Mean Absolute Error): **~0.65**.

### 3. Phát triển Ứng dụng Web
* Giao diện chia làm 2 luồng rõ ràng: **Gợi ý theo Nội dung** & **Gợi ý theo Cộng đồng**.
* Tích hợp bộ lọc **Mood (Tâm trạng)**: Map các thể loại phim với cảm xúc người dùng.
* Chức năng **Lịch sử (History)**: Cho phép người dùng chuyển đổi nguồn gợi ý giữa "Phim đang tìm" và "Phim vừa xem".

---

## 📂 IV. CẤU TRÚC DỰ ÁN

```text
MOVIE-AI-ULTIMATE/
│
├── app.py                   # 🚀 Source code chính (Streamlit Web App)
├── generate_bert.py         # Script chạy offline để tạo vector BERT
├── requirements.txt         # Danh sách thư viện cần cài đặt
│
├── data/
│   ├── movies_final_dataset.csv  # Dữ liệu phim đã làm sạch (+Poster)
│   ├── ratings.csv               # Dữ liệu đánh giá
│   └── bert_embeddings.npy       # File vector BERT đã tính toán trước
│
├── notebooks/               # Các file Jupyter Notebook phân tích
│   ├── 1_data_collection.ipynb
│   ├── 2_data_cleaning_eda.ipynb
│   ├── 3_recommendation_models.ipynb
│   └── 4_model_evaluation.ipynb
│
└── README.md                # Tài liệu dự án
````

-----

## 🚀 V. HƯỚNG DẪN CÀI ĐẶT (LOCAL)

Để chạy dự án trên máy cá nhân:

**Bước 1: Clone dự án**

```bash
git clone [LINK_GITHUB_CUA_BAN]
cd MOVIE-AI-ULTIMATE
```

**Bước 2: Cài đặt thư viện**

```bash
pip install -r requirements.txt
```

**Bước 3: Tạo vector BERT (Chạy 1 lần duy nhất)**

```bash
python generate_bert.py
# Bước này sẽ sinh ra file 'bert_embeddings.npy'
```

**Bước 4: Khởi chạy Web App**

```bash
streamlit run app.py
```

👉 *Truy cập: http://localhost:8501*

-----

## 📸 VI. DEMO SẢN PHẨM

**(Dán ảnh chụp màn hình giao diện Web Streamlit đẹp nhất của bạn vào đây)**

-----

## 📜 VII. KẾT LUẬN

Dự án đã xây dựng thành công một hệ thống gợi ý phim hoàn chỉnh, từ khâu xử lý dữ liệu thô đến việc triển khai một ứng dụng thông minh. Việc áp dụng **BERT** và **Context-aware** đã giúp hệ thống vượt trội hơn so với các phương pháp gợi ý truyền thống, mang lại trải nghiệm cá nhân hóa cao cho người dùng.

-----

*Project thực hiện bởi Nguyễn Thế Bình - 2025*

```
```