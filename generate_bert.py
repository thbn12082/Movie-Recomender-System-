import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. LOAD DỮ LIỆU
print("Đang đọc dữ liệu phim...")
movies = pd.read_csv('movies_final_dataset.csv')
movies['overview'] = movies['overview'].fillna('')

# 2. KHỞI TẠO MODEL BERT (Sentence-BERT)
# Model 'all-MiniLM-L6-v2' rất nhẹ và nhanh, chuyên dùng cho Semantic Search
print("Đang tải model BERT (lần đầu sẽ hơi lâu)...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 3. VECTOR HÓA (EMBEDDING)
# Bước này máy tính sẽ đọc hiểu nội dung 9000 phim và biến thành số
print("Đang chạy BERT để vector hóa nội dung phim (có thể mất 1-2 phút)...")
bert_embeddings = model.encode(movies['overview'].tolist(), show_progress_bar=True)

# 4. LƯU FILE KẾT QUẢ
# Lưu dưới dạng file NumPy (.npy) để đọc siêu nhanh
np.save('bert_embeddings.npy', bert_embeddings)

print("-" * 30)
print(f"XONG! Đã tạo file 'bert_embeddings.npy' với kích thước: {bert_embeddings.shape}")
print("Hãy upload file .npy này lên cùng thư mục dự án của bạn.")