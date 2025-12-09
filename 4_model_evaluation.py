import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

# 1. LOAD DỮ LIỆU
print("Đang tải và xử lý dữ liệu...")
try:
    ratings = pd.read_csv('ratings.csv')
except FileNotFoundError:
    ratings = pd.read_csv('ml-latest-small/ratings.csv')

# 2. CHUẨN BỊ MA TRẬN VỚI MEAN CENTERING (Quan trọng)
# Tạo ma trận pivot
user_movie_matrix = ratings.pivot_table(index='userId', columns='movieId', values='rating')

# Tính điểm trung bình của từng user (bỏ qua NaN)
user_means = user_movie_matrix.mean(axis=1)

# Trừ đi điểm trung bình (Centering)
# Lúc này: Điểm > 0 là thích, < 0 là ghét, = 0 là bình thường (hoặc chưa xem)
matrix_centered = user_movie_matrix.sub(user_means, axis=0).fillna(0)

X = matrix_centered.values

# 3. HUẤN LUYỆN MODEL SVD
# Sử dụng 20 đặc trưng tiềm ẩn
svd = TruncatedSVD(n_components=20, random_state=42)
X_transformed = svd.fit_transform(X)

# Tái tạo lại ma trận
X_reconstructed_centered = svd.inverse_transform(X_transformed)

# Cộng lại điểm trung bình đã trừ lúc đầu để ra điểm dự đoán thực tế
predicted_matrix = X_reconstructed_centered + user_means.values.reshape(-1, 1)

# 4. TÍNH TOÁN SAI SỐ (Chỉ tính trên những ô CÓ dữ liệu thực tế)
# Lấy ma trận gốc (chưa fillna) để biết ô nào là dữ liệu thật
original_matrix = user_movie_matrix.values

# Tạo mặt nạ (mask) cho những vị trí có dữ liệu (không phải NaN)
mask = ~np.isnan(original_matrix)

# Lấy dữ liệu thực và dự đoán tại các vị trí đó
actual = original_matrix[mask]
predicted = predicted_matrix[mask]

# Tính RMSE & MAE
mse = mean_squared_error(actual, predicted)
rmse = math.sqrt(mse)
mae = mean_absolute_error(actual, predicted)

print("-" * 40)
print(f"KẾT QUẢ ĐÁNH GIÁ (SAU KHI CẢI THIỆN):")
print(f"📉 RMSE (Sai số trung bình phương): {rmse:.4f}")
print(f"📉 MAE  (Sai số tuyệt đối):        {mae:.4f}")
print("-" * 40)

# Nhận xét
if rmse < 1.0:
    print("✅ TUYỆT VỜI! Sai số đã giảm xuống dưới 1.0. Mô hình hoạt động tốt.")
elif rmse < 1.5:
    print("⚠️ Khá ổn. Sai số chấp nhận được cho thuật toán SVD cơ bản.")
else:
    print("❌ Vẫn còn cao. Cần kiểm tra lại dữ liệu.")