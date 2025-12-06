import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. LOAD DỮ LIỆU
movies = pd.read_csv('movies_final_dataset.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv') # File gốc từ MovieLens

print(f"Số lượng phim ban đầu: {len(movies)}")

# 2. LÀM SẠCH (DATA CLEANING)
# Xóa các dòng trùng lặp (nếu có)
movies = movies.drop_duplicates(subset=['tmdbId'])

# Xử lý Missing Values: Nếu không có tóm tắt (overview), điền chuỗi rỗng
movies['overview'] = movies['overview'].fillna('')

# Lọc bỏ phim quá ít người vote (để tránh nhiễu dữ liệu cho Model)
# Chỉ giữ phim có trên 10 vote trên TMDB
movies_clean = movies[movies['vote_count'] > 10].copy()

print(f"Số lượng phim sau khi làm sạch: {len(movies_clean)}")

# 3. VECTOR HÓA TEXT (TF-IDF) - Yêu cầu nâng cao
# Biến cột 'overview' thành ma trận số để máy tính hiểu nội dung phim
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_clean['overview'])

print(f"Kích thước ma trận TF-IDF: {tfidf_matrix.shape}")
# Kết quả (số phim, số từ vựng) -> Ví dụ: (8000, 25000)


# VẼ BIỂU ĐỒ



# Cấu hình giao diện đẹp
sns.set_style("whitegrid")
plt.figure(figsize=(12, 5))

# BIỂU ĐỒ 1: Phân bố điểm đánh giá (Rating Distribution)
plt.subplot(1, 2, 1)
sns.histplot(movies_clean['vote_average'], bins=20, kde=True, color='skyblue')
plt.title('Phân bố điểm đánh giá trung bình (TMDB)')
plt.xlabel('Điểm số (0-10)')
plt.ylabel('Số lượng phim')

# BIỂU ĐỒ 2: Top 10 Phim phổ biến nhất
plt.subplot(1, 2, 2)
top_movies = movies_clean.sort_values(by='vote_count', ascending=False).head(10)
# Sửa dòng này:
sns.barplot(x='vote_count', y='title', data=top_movies, palette='viridis', hue='title', legend=False)
plt.title('Top 10 Phim có lượt vote cao nhất')
plt.xlabel('Lượt vote')
plt.ylabel('')

plt.tight_layout()
plt.show()