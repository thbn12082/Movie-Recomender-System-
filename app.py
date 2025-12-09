import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Movie Pro - All in One", layout="wide")

# ==========================================
# MODULE QUẢN LÝ FILE JSON
# ==========================================
USER_DATA_FILE = 'user_favorites.json'


def load_favorites_from_disk():
    if os.path.exists(USER_DATA_FILE):
        try:
            with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []


def save_favorites_to_disk(fav_list):
    with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(fav_list, f, ensure_ascii=False)


# ==========================================
# KHỞI TẠO STATE
# ==========================================
if 'favorites' not in st.session_state:
    st.session_state['favorites'] = load_favorites_from_disk()

if 'selected_movie_name' not in st.session_state:
    st.session_state['selected_movie_name'] = ''

if 'rec_mode' not in st.session_state:
    st.session_state['rec_mode'] = "✨ Dựa trên Tủ phim của tôi"


# ==========================================
# LOAD DATA & TRAIN MODELS
# ==========================================
@st.cache_data
def load_data():
    movies = pd.read_csv('movies_final_dataset.csv')
    movies['overview'] = movies['overview'].fillna('')
    try:
        ratings = pd.read_csv('ratings.csv')
    except:
        ratings = pd.read_csv('ml-latest-small/ratings.csv')
    return movies, ratings


@st.cache_resource
def train_models(movies, ratings):
    # Content-Based
    try:
        bert_matrix = np.load('bert_embeddings.npy')
        cosine_sim = cosine_similarity(bert_matrix, bert_matrix)
    except:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies['overview'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Collaborative
    user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
    SVD = TruncatedSVD(n_components=20, random_state=42)
    matrix_reduced = SVD.fit_transform(user_movie_matrix.T)
    corr_mat = np.corrcoef(matrix_reduced)

    return cosine_sim, user_movie_matrix, corr_mat


try:
    movies, ratings = load_data()
    if st.session_state['selected_movie_name'] == '':
        st.session_state['selected_movie_name'] = movies['title'].values[0]
    cosine_sim, user_movie_matrix, corr_mat = train_models(movies, ratings)
except Exception as e:
    st.error(f"Lỗi load data: {e}")
    st.stop()


# ==========================================
# CÁC HÀM CALLBACK
# ==========================================
def navigate_to_movie(movie_title):
    st.session_state['selected_movie_name'] = movie_title
    st.session_state['rec_mode'] = "🔍 Tìm kiếm phim lẻ"


def add_to_favorites(movie_title):
    if movie_title not in st.session_state['favorites']:
        st.session_state['favorites'].append(movie_title)
        save_favorites_to_disk(st.session_state['favorites'])
        st.toast(f"Đã thêm '{movie_title}' vào tủ!", icon="❤️")


def remove_from_favorites(movie_title):
    if movie_title in st.session_state['favorites']:
        st.session_state['favorites'].remove(movie_title)
        save_favorites_to_disk(st.session_state['favorites'])


# ==========================================
# LOGIC GỢI Ý
# ==========================================
def filter_by_mood(movies_df, mood):
    mood_genre_map = {
        "Vui vẻ": ["Comedy", "Animation", "Family", "Music"],
        "Buồn": ["Drama", "Romance", "Documentary"],
        "Hồi hộp": ["Action", "Thriller", "Adventure", "Crime"],
        "Sợ hãi": ["Horror", "Mystery"],
        "Bình thường": []
    }
    target_genres = mood_genre_map.get(mood, [])
    if not target_genres: return movies_df
    pattern = '|'.join(target_genres)
    return movies_df[movies_df['genres'].str.contains(pattern, case=False, na=False)]


def get_single_recommendations(title, mood):
    idx_list = movies.index[movies['title'] == title].tolist()
    if not idx_list: return pd.DataFrame()
    idx = idx_list[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:30]
    movie_indices = [i[0] for i in sim_scores]
    results = movies.iloc[movie_indices].copy()
    results = filter_by_mood(results, mood)
    return results.head(10)


def get_collab_recs(title, user_movie_matrix, corr_mat, mood):
    movie_id_list = movies[movies['title'] == title]['movieId'].values
    if len(movie_id_list) == 0: return None
    movie_id = movie_id_list[0]
    if movie_id not in user_movie_matrix.columns: return None
    col_idx = user_movie_matrix.columns.get_loc(movie_id)
    corr_specific = corr_mat[col_idx]
    result = pd.DataFrame({'corr_score': corr_specific})
    result['movieId'] = user_movie_matrix.columns
    result = pd.merge(result, movies, on='movieId')
    result = result.sort_values('corr_score', ascending=False).iloc[1:]
    result = filter_by_mood(result, mood)
    return result.head(10)


def get_aggregated_recommendations(favorites_list, mood):
    if not favorites_list: return pd.DataFrame()
    movie_scores = {}
    for fav_movie in favorites_list:
        idx_list = movies.index[movies['title'] == fav_movie].tolist()
        if not idx_list: continue
        idx = idx_list[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        top_similar = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:21]
        for i, score in top_similar:
            title = movies.iloc[i]['title']
            if title in movie_scores:
                movie_scores[title] += score
            else:
                movie_scores[title] = score
    for fav in favorites_list:
        if fav in movie_scores: del movie_scores[fav]
    sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
    top_titles = [m[0] for m in sorted_movies[:30]]
    results = movies[movies['title'].isin(top_titles)]
    results = filter_by_mood(results, mood)
    return results.head(12)


# ==========================================
# GIAO DIỆN (UI)
# ==========================================

with st.sidebar:
    st.title("🍿 Menu")
    # THAY ĐỔI: Thêm mục "Biểu đồ phân tích" vào Menu
    page = st.radio("Chọn trang:", ["🏠 Trang chủ", "⚙️ Quản lý Tủ phim", "📊 Biểu đồ phân tích"])
    st.divider()

    if page == "🏠 Trang chủ":
        st.header("🔍 Cấu hình gợi ý")
        recommendation_mode = st.radio(
            "Nguồn gợi ý:",
            ["✨ Dựa trên Tủ phim của tôi", "🔍 Tìm kiếm phim lẻ"],
            key='rec_mode'
        )

        if recommendation_mode == "🔍 Tìm kiếm phim lẻ":
            selected_movie = st.selectbox("Chọn phim:", movies['title'].values, key='selected_movie_name')
        else:
            if not st.session_state['favorites']:
                st.info("Tủ phim đang trống.")
            else:
                st.success(f"Đang dùng {len(st.session_state['favorites'])} phim để phân tích.")

        st.divider()
        selected_mood = st.selectbox("Tâm trạng:", ["Bình thường", "Vui vẻ", "Buồn", "Hồi hộp", "Sợ hãi"])

# --- TRANG 1: TRANG CHỦ ---
if page == "🏠 Trang chủ":
    if recommendation_mode == "✨ Dựa trên Tủ phim của tôi":
        st.title("✨ Gợi ý dành riêng cho BẠN")
        fav_list = st.session_state['favorites']

        if not fav_list:
            st.warning("⚠️ Tủ phim của bạn đang trống!")
            st.info("👉 Hãy chuyển sang chế độ **'🔍 Tìm kiếm phim lẻ'** để thêm phim.")
        else:
            with st.spinner("AI đang phân tích gu điện ảnh của bạn..."):
                aggregated_recs = get_aggregated_recommendations(fav_list, selected_mood)

            if not aggregated_recs.empty:
                st.success(f"Gợi ý phù hợp với tâm trạng **'{selected_mood}'**:")
                for i, row in enumerate(aggregated_recs.iterrows()):
                    if i % 4 == 0: cols = st.columns(4)
                    with cols[i % 4]:
                        m = row[1]
                        if pd.notna(m['poster_path']):
                            st.image(f"https://image.tmdb.org/t/p/w300{m['poster_path']}")
                        st.subheader(m['title'])

                        if m['title'] not in st.session_state['favorites']:
                            st.button("❤️ Thêm", key=f"agg_add_{m['title']}", on_click=add_to_favorites,
                                      args=(m['title'],))
                        else:
                            st.write("✅ Đã thích")

                        st.button("Chi tiết", key=f"agg_view_{m['title']}", on_click=navigate_to_movie,
                                  args=(m['title'],))
            else:
                st.warning("Không tìm thấy phim phù hợp.")

    else:
        target_movie = st.session_state['selected_movie_name']
        st.title(f"🎬 Khám phá: {target_movie}")

        movie_info = movies[movies['title'] == target_movie].iloc[0]
        c1, c2 = st.columns([1, 3])
        with c1:
            if pd.notna(movie_info['poster_path']):
                st.image(f"https://image.tmdb.org/t/p/w500{movie_info['poster_path']}")
        with c2:
            st.write(f"**Thể loại:** {movie_info['genres']}")
            st.write(movie_info['overview'][:500])
            if target_movie not in st.session_state['favorites']:
                st.button("❤️ Thêm vào Tủ phim", on_click=add_to_favorites, args=(target_movie,))
            else:
                st.success("✅ Đã có trong tủ!")

        st.divider()
        t1, t2 = st.tabs(["Nội dung tương tự", "Cộng đồng cũng thích"])


        def show_movie_grid(results, key_prefix):
            if results is not None and not results.empty:
                for i, row in enumerate(results.iterrows()):
                    if i % 5 == 0: cols = st.columns(5)
                    with cols[i % 5]:
                        m = row[1]
                        if pd.notna(m['poster_path']):
                            st.image(f"https://image.tmdb.org/t/p/w200{m['poster_path']}")
                        st.write(f"**{m['title']}**")
                        st.button("👉 Xem", key=f"{key_prefix}_{m['title']}", on_click=navigate_to_movie,
                                  args=(m['title'],))
            else:
                st.warning("Không tìm thấy phim phù hợp.")


        with t1:
            res = get_single_recommendations(target_movie, selected_mood)
            show_movie_grid(res, "content")
        with t2:
            res = get_collab_recs(target_movie, user_movie_matrix, corr_mat, selected_mood)
            show_movie_grid(res, "collab")

# --- TRANG 2: QUẢN LÝ TỦ PHIM ---
elif page == "⚙️ Quản lý Tủ phim":
    st.title("⚙️ Quản lý Tủ phim")
    fav_list = st.session_state['favorites']
    if fav_list:
        st.write(f"Bạn đang lưu **{len(fav_list)}** phim yêu thích.")
        st.divider()
        for i, title in enumerate(fav_list):
            if i % 4 == 0: cols = st.columns(4)
            with cols[i % 4]:
                movie_info = movies[movies['title'] == title]
                if not movie_info.empty:
                    m = movie_info.iloc[0]
                    if pd.notna(m['poster_path']):
                        st.image(f"https://image.tmdb.org/t/p/w200{m['poster_path']}")
                    else:
                        st.image("https://via.placeholder.com/200x300?text=No+Image")
                    st.write(f"**{title}**")
                    if st.button("🗑️ Xóa", key=f"del_grid_{title}"):
                        remove_from_favorites(title)
                        st.rerun()
        st.divider()
        if st.button("Xóa sạch tủ phim", type="primary"):
            st.session_state['favorites'] = []
            save_favorites_to_disk([])
            st.rerun()
    else:
        st.info("Tủ phim hiện đang trống.")

# --- TRANG 3: BIỂU ĐỒ PHÂN TÍCH (EDA) ---
elif page == "📊 Biểu đồ phân tích":
    st.title("📊 Phân tích Dữ liệu (EDA Dashboard)")
    st.markdown("Tổng quan về bộ dữ liệu phim và hành vi người dùng.")

    # Chuẩn bị dữ liệu sạch cho biểu đồ (giống file 2_data_cleaning_eda.py)
    movies_clean = movies[movies['vote_count'] > 10].copy()

    # Vẽ biểu đồ bằng Matplotlib/Seaborn
    sns.set_style("whitegrid")

    # Tạo Tabs để xem từng biểu đồ cho rõ (Web xem Grid nhỏ quá khó nhìn)
    tab_eda1, tab_eda2 = st.tabs(["📈 Thống kê cơ bản", "☁️ WordCloud & Tương quan"])

    with tab_eda1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("1. Phân bố điểm đánh giá")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.histplot(movies_clean['vote_average'], bins=20, kde=True, color='#3498db', ax=ax1)
            ax1.set_xlabel('Điểm số')
            st.pyplot(fig1)

        with c2:
            st.subheader("2. Top 10 Phim phổ biến nhất")
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            top_movies = movies_clean.sort_values(by='vote_count', ascending=False).head(10)
            sns.barplot(x='vote_count', y='title', data=top_movies, palette='viridis', hue='title', legend=False,
                        ax=ax2)
            ax2.set_ylabel('')
            st.pyplot(fig2)

        st.subheader("3. Top Thể loại phim")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        if 'genres' in movies_clean.columns:
            genres_exploded = movies_clean['genres'].str.split('|').explode()
            top_genres = genres_exploded.value_counts().head(10).index
            genres_filtered = genres_exploded[genres_exploded.isin(top_genres)]
            sns.countplot(y=genres_filtered, order=top_genres, palette='muted', hue=genres_filtered, legend=False,
                          ax=ax3)
            ax3.set_ylabel('')
            st.pyplot(fig3)

    with tab_eda2:
        col_heat, col_cloud = st.columns(2)

        with col_heat:
            st.subheader("4. Heatmap tương quan")
            fig4, ax4 = plt.subplots(figsize=(6, 5))
            corr_cols = ['vote_average', 'vote_count']
            if set(corr_cols).issubset(movies_clean.columns):
                correlation_matrix = movies_clean[corr_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax4)
                st.pyplot(fig4)

        with col_cloud:
            st.subheader("5. WordCloud Nội dung")
            with st.spinner("Đang tạo WordCloud..."):
                text = " ".join(str(review) for review in movies_clean.overview.fillna(''))
                wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(text)
                fig5, ax5 = plt.subplots(figsize=(8, 5))
                ax5.imshow(wordcloud, interpolation='bilinear')
                ax5.axis("off")
                st.pyplot(fig5)