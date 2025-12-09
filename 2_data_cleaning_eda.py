import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Cấu hình hiển thị
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")

print("--- BẮT ĐẦU QUÁ TRÌNH LÀM SẠCH VÀ EDA ---")

# 1. LOAD DỮ LIỆU
try:
    movies = pd.read_csv('movies_final_dataset.csv')
    print(f"✅ Đã tải dữ liệu gốc: {movies.shape}")
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file 'movies_final_dataset.csv'.")
    exit()

# ==============================================================================
# PHẦN 1: LÀM SẠCH DỮ LIỆU (DATA CLEANING) - Thực hiện đủ 5/5 yêu cầu
# ==============================================================================

print("\n>>> ĐANG THỰC HIỆN 5 TÁC VỤ LÀM SẠCH:")

# 1. Loại bỏ Duplicate (Trùng lặp)
# Đề bài: "Loại bỏ duplicate"
initial_count = len(movies)
movies.drop_duplicates(subset=['tmdbId'], inplace=True)
print(f"1. [Duplicate] Đã xóa {initial_count - len(movies)} phim trùng lặp.")

# 2. Xử lý Missing Values (Giá trị thiếu)
# Đề bài: "Missing values"
print(f"2. [Missing Values] Số dòng thiếu overview trước khi xử lý: {movies['overview'].isnull().sum()}")
movies['overview'] = movies['overview'].fillna('')
print(f"   -> Đã điền xong. Số dòng thiếu hiện tại: {movies['overview'].isnull().sum()}")

# 3. Xử lý Outlier (Giá trị nhiễu)
# Đề bài: "Xử lý outlier"
# Lọc bỏ những phim có quá ít lượt vote (dưới 10) vì số liệu này không đáng tin cậy
movies_clean = movies[movies['vote_count'] > 10].copy()
print(f"3. [Outlier] Đã lọc bỏ các phim rác (vote < 10). Số lượng phim còn lại: {len(movies_clean)}")

# 4. Chuẩn hóa dữ liệu (Normalization)
# Đề bài: "Chuẩn hóa dữ liệu"
# Đưa điểm đánh giá (0-10) về thang đo (0-1) để đồng bộ dữ liệu
scaler = MinMaxScaler()
movies_clean['vote_average_normalized'] = scaler.fit_transform(movies_clean[['vote_average']])
print("4. [Normalization] Đã chuẩn hóa cột 'vote_average' về thang [0-1].")

# 5. Vector hóa văn bản (Text Vectorization)
# Đề bài: "Vector hóa (TF-IDF, BOW...)"
print("5. [Vectorization] Đang thực hiện TF-IDF cho cột overview...")
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)  # Lấy 5000 từ quan trọng nhất
tfidf_matrix = tfidf.fit_transform(movies_clean['overview'])
print(f"   -> Kết quả ma trận TF-IDF: {tfidf_matrix.shape}")

print("\n✅ HOÀN TẤT LÀM SẠCH DỮ LIỆU!")

# ==============================================================================
# PHẦN 2: TRỰC QUAN HÓA DỮ LIỆU (EDA) - Thực hiện đủ 5 biểu đồ
# ==============================================================================

print("\n>>> ĐANG VẼ BIỂU ĐỒ TRỰC QUAN HÓA...")

# Thiết lập khung hình lớn (Grid 2 dòng x 3 cột)
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('BÁO CÁO PHÂN TÍCH DỮ LIỆU PHIM (EDA)', fontsize=24, fontweight='bold', color='navy')

# --- BIỂU ĐỒ 1: PHÂN BỐ RATING (Histogram) ---
# Đề bài: "Phân bố rating"
sns.histplot(movies_clean['vote_average'], bins=20, kde=True, color='#3498db', ax=axes[0, 0])
axes[0, 0].set_title('1. Phân bố điểm đánh giá (0-10)', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Điểm số')
axes[0, 0].set_ylabel('Số lượng phim')

# --- BIỂU ĐỒ 2: TOP ITEMS (Bar Chart) ---
# Đề bài: "Top items"
top_movies = movies_clean.sort_values(by='vote_count', ascending=False).head(10)
sns.barplot(x='vote_count', y='title', data=top_movies, palette='viridis', hue='title', legend=False, ax=axes[0, 1])
axes[0, 1].set_title('2. Top 10 Phim có lượt tương tác cao nhất', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Lượt vote')
axes[0, 1].set_ylabel('')

# --- BIỂU ĐỒ 3: TẦN SUẤT NHÓM SẢN PHẨM (Bar Chart) ---
# Đề bài: "Tần suất nhóm sản phẩm"
if 'genres' in movies_clean.columns:
    genres_exploded = movies_clean['genres'].str.split('|').explode()
    top_genres = genres_exploded.value_counts().head(10).index
    genres_filtered = genres_exploded[genres_exploded.isin(top_genres)]

    sns.countplot(y=genres_filtered, order=top_genres, palette='muted', hue=genres_filtered, legend=False,
                  ax=axes[0, 2])
    axes[0, 2].set_title('3. Top 10 Thể loại phim phổ biến nhất', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('Số lượng phim')
    axes[0, 2].set_ylabel('')

# --- BIỂU ĐỒ 4: HEATMAP (Heatmap) ---
# Đề bài: "Heatmap, bar chart, histogram"
# Tính tương quan giữa các biến số
corr_cols = ['vote_average', 'vote_count', 'vote_average_normalized']
if set(corr_cols).issubset(movies_clean.columns):
    correlation_matrix = movies_clean[corr_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=axes[1, 0])
    axes[1, 0].set_title('4. Heatmap tương quan các chỉ số', fontsize=14, fontweight='bold')

# --- BIỂU ĐỒ 5: WORDCLOUD (Nâng cao) ---
# Trực quan hóa dữ liệu văn bản
text = " ".join(str(review) for review in movies_clean.overview.fillna(''))
# Tạo wordcloud
wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(text)

# Hiển thị vào ô giữa dòng dưới (Vị trí 1,1)
axes[1, 1].imshow(wordcloud, interpolation='bilinear')
axes[1, 1].axis("off")
axes[1, 1].set_title('5. WordCloud: Từ khóa trong nội dung phim', fontsize=14, fontweight='bold')

# Ẩn ô cuối cùng (Dòng 1, Cột 2) cho đẹp đội hình
axes[1, 2].axis('off')
axes[1, 2].text(0.5, 0.5, 'DATA SCIENCE PROJECT\nMovies Recommendation System',
                ha='center', va='center', fontsize=20, color='gray', alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Chừa chỗ cho Title lớn ở trên
plt.show()

print("\n>>> ĐÃ HOÀN THÀNH TẤT CẢ YÊU CẦU!")