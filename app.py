import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import TruncatedSVD

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Movie Recommender System", layout="wide")


# --- 1. HÀM LOAD DỮ LIỆU (Dùng Cache để web chạy nhanh) ---
@st.cache_data
def load_data():
    # Đọc dữ liệu phim đã xử lý (có poster và overview)
    movies = pd.read_csv('movies_final_dataset.csv')
    movies['overview'] = movies['overview'].fillna('')

    # Đọc dữ liệu rating
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    return movies, ratings


# --- 2. HUẤN LUYỆN MODEL (Chạy ngầm 1 lần khi mở web) ---
@st.cache_resource
def train_models(movies, ratings):
    # A. Content-Based (TF-IDF)
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # B. Collaborative Filtering (SVD)
    user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    SVD = TruncatedSVD(n_components=20, random_state=42)
    matrix_reduced = SVD.fit_transform(user_movie_matrix.T)
    corr_mat = np.corrcoef(matrix_reduced)

    return cosine_sim, user_movie_matrix, corr_mat


# Load dữ liệu và Train model
try:
    data_load_state = st.text('Đang tải dữ liệu và huấn luyện AI... vui lòng chờ...')
    movies, ratings = load_data()
    cosine_sim, user_movie_matrix, corr_mat = train_models(movies, ratings)
    data_load_state.text('Hệ thống đã sẵn sàng!')
except Exception as e:
    st.error(f"Lỗi: Không tìm thấy file dữ liệu. Hãy chắc chắn file csv nằm cùng thư mục. Chi tiết: {e}")
    st.stop()


# --- 3. CÁC HÀM GỢI Ý ---
def get_content_recs(title, movies, cosine_sim):
    idx = movies.index[movies['title'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # Lấy 5 phim
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices]


def get_collab_recs(title, movies, user_movie_matrix, corr_mat):
    movie_id = movies[movies['title'] == title]['movieId'].values[0]
    if movie_id not in user_movie_matrix.columns:
        return None
    col_idx = user_movie_matrix.columns.get_loc(movie_id)
    corr_specific = corr_mat[col_idx]

    result = pd.DataFrame({'corr_score': corr_specific})
    result['movieId'] = user_movie_matrix.columns
    result = pd.merge(result, movies, on='movieId')
    result = result.sort_values('corr_score', ascending=False).head(6).iloc[1:]
    return result


# --- 4. GIAO DIỆN NGƯỜI DÙNG (UI) ---
st.title("🎬 Hệ thống Gợi ý Phim Thông minh")
st.markdown("Đồ án cuối kỳ môn Khoa học Dữ liệu - **Hybrid Recommendation System**")

# Sidebar chọn phim
st.sidebar.header("🔍 Tìm kiếm phim")
selected_movie = st.sidebar.selectbox(
    "Nhập tên phim bạn thích:",
    movies['title'].values
)

# Hiển thị phim đang chọn
if selected_movie:
    movie_info = movies[movies['title'] == selected_movie].iloc[0]

    col1, col2 = st.columns([1, 2])
    with col1:
        # Hiển thị Poster (Lấy từ link TMDB)
        if pd.notna(movie_info['poster_path']):
            full_poster_url = f"https://image.tmdb.org/t/p/w500{movie_info['poster_path']}"
            st.image(full_poster_url, width=200)
        else:
            st.write("(Không có poster)")
    with col2:
        st.subheader(movie_info['title'])
        st.write(f"**Thể loại:** {movie_info['genres']}")
        st.write(f"**Đánh giá:** ⭐ {movie_info['vote_average']}/10")
        st.write(f"**Nội dung:** {movie_info['overview']}")

    st.markdown("---")

    # Hiển thị 2 luồng gợi ý
    st.header(f"Gợi ý cho bạn vì bạn thích: {selected_movie}")

    tab1, tab2 = st.tabs(["🧩 Gợi ý theo Nội dung (Content-Based)", "👥 Gợi ý theo Cộng đồng (Collaborative)"])

    # TAB 1: CONTENT BASED
    with tab1:
        st.caption("Các phim có cốt truyện tương tự.")
        results = get_content_recs(selected_movie, movies, cosine_sim)

        cols = st.columns(5)
        for i, row in enumerate(results.iterrows()):
            with cols[i]:
                movie = row[1]
                st.write(f"**{movie['title']}**")
                if pd.notna(movie['poster_path']):
                    st.image(f"https://image.tmdb.org/t/p/w200{movie['poster_path']}")

    # TAB 2: COLLABORATIVE FILTERING
    with tab2:
        st.caption("Các phim mà người hâm mộ phim này cũng thích.")
        results_collab = get_collab_recs(selected_movie, movies, user_movie_matrix, corr_mat)

        if results_collab is not None:
            cols = st.columns(5)
            for i, row in enumerate(results_collab.iterrows()):
                with cols[i]:
                    movie = row[1]
                    st.write(f"**{movie['title']}**")
                    if pd.notna(movie['poster_path']):
                        st.image(f"https://image.tmdb.org/t/p/w200{movie['poster_path']}")
        else:
            st.warning("Phim này chưa có đủ dữ liệu rating để gợi ý theo cộng đồng.")