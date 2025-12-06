import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# 1. LOAD LẠI DỮ LIỆU ĐÃ SẠCH
movies = pd.read_csv('movies_final_dataset.csv')
movies['overview'] = movies['overview'].fillna('')

# 2. TẠO MA TRẬN TƯƠNG ĐỒNG (COSINE SIMILARITY)
# Bước này tính độ giống nhau giữa 9000 phim với nhau.
# Máy tính sẽ so sánh từng từ trong 'overview'
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['overview'])

print("Đang tính toán độ tương đồng... (Chờ chút nhé)")
# Dùng linear_kernel để tính nhanh hơn cosine_similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# 3. HÀM GỢI Ý PHIM
# Tạo bảng tra cứu: Tên phim -> Vị trí trong bảng (Index)
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()


def get_content_recommendations(title, cosine_sim=cosine_sim):
    # Kiểm tra xem phim có trong data không
    if title not in indices:
        return "Phim không tồn tại trong hệ thống!"

    # Lấy index của phim
    idx = indices[title]

    # Lấy danh sách độ giống nhau của phim này với tất cả phim khác
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sắp xếp giảm dần (phim giống nhất ở trên đầu)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Lấy 10 phim giống nhất (bỏ phim đầu tiên vì là chính nó)
    sim_scores = sim_scores[1:11]

    # Lấy tên phim trả về
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]


# Hàm tìm tên phim chuẩn từ từ khóa (VD: gõ "Toy Story" -> tìm ra "Toy Story (1995)")
def find_movie_title(keyword):
    # Tìm tất cả phim có chứa từ khóa (không phân biệt hoa thường)
    results = movies[movies['title'].str.contains(keyword, case=False)]
    if not results.empty:
        # Trả về tên phim đầu tiên tìm thấy
        return results.iloc[0]['title']
    return None

# --- CHẠY THỬ LẠI ---
keyword = "Toy Story"
correct_title = find_movie_title(keyword)

if correct_title:
    print(f"Đang gợi ý cho phim: '{correct_title}'")
    print(get_content_recommendations(correct_title))
else:
    print("Không tìm thấy phim nào khớp với từ khóa!")

# --- PHẦN 2: COLLABORATIVE FILTERING (Matrix Factorization với Sklearn) ---
# Thay thế cho Surprise, không cần cài đặt gì thêm

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD

# 1. CHUẨN BỊ DỮ LIỆU
print("Đang xử lý dữ liệu Rating...")
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Tạo Ma trận User-Item (Hàng là User, Cột là Phim, Giá trị là điểm rating)
# Fillna(0) nghĩa là chưa xem thì coi như 0 điểm
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

print(f"Kích thước ma trận: {user_movie_matrix.shape}")
# (610 users, 9724 movies)

# 2. XÂY DỰNG MODEL SVD (Matrix Factorization)
# Chúng ta nén dữ liệu từ 9700 cột xuống còn 20 đặc trưng tiềm ẩn (latent features)
# Đây là kỹ thuật giảm chiều dữ liệu, giúp máy hiểu "gu" ngầm của user
SVD = TruncatedSVD(n_components=20, random_state=42)
matrix_reduced = SVD.fit_transform(user_movie_matrix.T)  # Transpose để tính tương đồng giữa các Phim

print("Đang tính toán độ tương đồng giữa các phim dựa trên lịch sử user...")
# Tính ma trận tương đồng (Correlation Matrix)
corr_mat = np.corrcoef(matrix_reduced)


# 3. HÀM GỢI Ý (COLLABORATIVE)
# Logic: Nếu user thích phim A, tìm các phim có vector rating tương tự phim A
def get_collaborative_recommendations(movie_title):
    # Tìm ID của phim từ tên
    movie_list = movies[movies['title'] == movie_title]
    if len(movie_list) == 0:
        return "Không tìm thấy phim này!"

    movie_id = movie_list.iloc[0]['movieId']

    # Kiểm tra xem phim có trong ma trận rating không
    if movie_id not in user_movie_matrix.columns:
        return "Chưa có ai chấm điểm phim này nên không thể gợi ý theo Collaborative!"

    # Lấy vị trí index của phim trong ma trận
    col_idx = user_movie_matrix.columns.get_loc(movie_id)

    # Lấy các phim tương đồng nhất (trong corr_mat)
    corr_specific = corr_mat[col_idx]

    # Tạo DataFrame kết quả
    result = pd.DataFrame({'corr_score': corr_specific})

    # Gán lại tên phim để hiển thị cho đẹp
    # (Mẹo: lấy index của columns gán vào)
    result['movieId'] = user_movie_matrix.columns
    result = pd.merge(result, movies[['movieId', 'title']], on='movieId')

    # Sắp xếp và lấy top 10 (bỏ cái đầu tiên là chính nó)
    result = result.sort_values('corr_score', ascending=False).head(11).iloc[1:]

    return result[['title', 'corr_score']]


# --- TEST THỬ ---
print("-" * 30)
target_movie = "Toy Story (1995)"
print(f"Gợi ý phim cho người thích '{target_movie}' (Dựa trên hành vi người dùng):")
print(get_collaborative_recommendations(target_movie))