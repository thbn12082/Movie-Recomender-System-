import pandas as pd
import requests
import time
from tqdm import tqdm  # Thư viện này để hiện thanh tiến trình cho đẹp

# 1. CẤU HÌNH
# Thay 'YOUR_TMDB_API_KEY' bằng key bạn vừa lấy ở Bước 2
API_KEY = 'd6715fe43ae18c370d0aeb56d88b44f6'


# 2. ĐỌC DỮ LIỆU MOVIELENS
# Đọc file links để lấy tmdbId
links_df = pd.read_csv('ml-latest-small/links.csv')
# Đọc file movies để lấy tên gốc
movies_df = pd.read_csv('ml-latest-small/movies.csv')

# Loại bỏ các dòng không có tmdbId (bị NaN)
links_df = links_df.dropna(subset=['tmdbId'])
links_df['tmdbId'] = links_df['tmdbId'].astype(int)

# Merge 2 bảng lại để có đủ thông tin id
full_df = pd.merge(movies_df, links_df, on='movieId')

print(f"Tổng số phim cần lấy thông tin: {len(full_df)}")


# 3. HÀM GỌI API
def get_movie_details(tmdb_id):
    url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={API_KEY}&language=en-US"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                'tmdbId': tmdb_id,
                'overview': data.get('overview', ''),  # Lấy tóm tắt phim (để làm NLP)
                'poster_path': data.get('poster_path', ''),  # Lấy đường dẫn ảnh
                'release_date': data.get('release_date', ''),
                'vote_average': data.get('vote_average', 0),
                'vote_count': data.get('vote_count', 0)
            }
    except Exception as e:
        print(f"Lỗi tại ID {tmdb_id}: {e}")
    return None


# 4. CHẠY VÒNG LẶP (Crawl Data)
enriched_data = []

# Chỉ chạy thử 5 phim đầu tiên để test trước khi chạy hết (xóa .head(5) để chạy full)
# Lưu ý: Chạy full 9000 phim sẽ mất khoảng 30-45 phút tùy mạng
print("Đang bắt đầu thu thập dữ liệu...")

for tmdb_id in tqdm(full_df['tmdbId']):
    details = get_movie_details(tmdb_id)
    if details:
        enriched_data.append(details)

    # Sleep 0.1s để không bị TMDB chặn vì spam request
    time.sleep(0.1)

# 5. LƯU KẾT QUẢ
details_df = pd.DataFrame(enriched_data)

# Ghép dữ liệu vừa lấy vào bảng gốc
final_df = pd.merge(full_df, details_df, on='tmdbId', how='left')

# Lưu ra file CSV mới để dùng cho các bước sau
final_df.to_csv('movies_final_dataset.csv', index=False)

print("Hoàn tất! Đã lưu file 'movies_final_dataset.csv'")
print(final_df.head())