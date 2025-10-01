import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# 1. Tải dữ liệu
# Đảm bảo file 'HR_comma_sep.csv' nằm cùng thư mục với file code,
# hoặc cung cấp đường dẫn đầy đủ đến file.
try:
    df = pd.read_csv('HR_comma_sep.csv')
except FileNotFoundError:
    print("Không tìm thấy file 'HR_comma_sep.csv'. Vui lòng kiểm tra lại đường dẫn.")
    exit()

# 2. Tiền xử lý dữ liệu
# Chuyển đổi các cột dạng chữ sang dạng số
le = LabelEncoder()
df['salary'] = le.fit_transform(df['salary'])
df = pd.get_dummies(df, columns=['Department'], drop_first=True)

# 3. Chia dữ liệu
# Xác định biến mục tiêu (y) và các biến dự báo (X)
X = df.drop('left', axis=1)
y = df['left']

# Chia thành tập huấn luyện (70%) và tập kiểm thử (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Huấn luyện và Đánh giá các mô hình

# --- Cây Quyết Định ---
print("--- 1. Cây Quyết Định (Decision Tree) ---")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
print(f"Độ chính xác: {accuracy_score(y_test, dt_preds):.4f}")
print("Báo cáo phân loại:")
print(classification_report(y_test, dt_preds))

# --- K-Hàng xóm gần nhất (KNN) ---
print("\n--- 2. K-Hàng xóm gần nhất (KNN) ---")
knn_model = KNeighborsClassifier()
knn_model.fit(X_train, y_train)
knn_preds = knn_model.predict(X_test)
print(f"Độ chính xác: {accuracy_score(y_test, knn_preds):.4f}")
print("Báo cáo phân loại:")
print(classification_report(y_test, knn_preds))

# --- Máy Vector Hỗ trợ (SVM) ---
print("\n--- 3. Máy Vector Hỗ trợ (SVM) ---")
svm_model = SVC(random_state=42)
svm_model.fit(X_train, y_train)
svm_preds = svm_model.predict(X_test)
print(f"Độ chính xác: {accuracy_score(y_test, svm_preds):.4f}")
print("Báo cáo phân loại:")
print(classification_report(y_test, svm_preds))