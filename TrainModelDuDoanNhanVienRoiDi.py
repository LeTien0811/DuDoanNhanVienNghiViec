import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

print("Bắt đầu quá trình huấn luyện mô hình...")

# --- 1. Tải và xử lý dữ liệu ---
# Đảm bảo file 'HR_comma_sep.csv' nằm cùng thư mục với file này
try:
    df = pd.read_csv('HR_comma_sep.csv')
    print("-> Đã tải dữ liệu thành công.")
except FileNotFoundError:
    print("LỖI: Không tìm thấy file 'HR_comma_sep.csv'. Vui lòng kiểm tra lại.")
    exit()

# Chuyển đổi các cột chữ thành số
df = pd.get_dummies(df, columns=['Department', 'salary'], drop_first=True)

# --- 2. Chuẩn bị dữ liệu cho mô hình ---
X = df.drop('left', axis=1)
y = df['left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("-> Đã chuẩn bị và chia dữ liệu xong.")

# --- 3. Xây dựng và huấn luyện mô hình Random Forest ---
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("-> Đã huấn luyện mô hình thành công.")

# --- 4. Lưu lại mô hình đã huấn luyện ---
# Lệnh này sẽ tạo ra file .pkl trong chính thư mục bạn đang chạy
joblib.dump(model, 'employee_churn_model.pkl')

print("\n----------------------------------------------------")
print("✅ HOÀN TẤT! Đã lưu mô hình vào file 'employee_churn_model.pkl'")
print("Bây giờ bạn có thể kiểm tra thư mục dự án của mình.")
print("----------------------------------------------------")