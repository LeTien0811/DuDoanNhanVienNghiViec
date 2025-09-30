import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Tải và xử lý dữ liệu ---
df = pd.read_csv('HR_comma_sep.csv')
df = pd.get_dummies(df, columns=['Department', 'salary'], drop_first=True)

# --- 2. Chuẩn bị dữ liệu cho mô hình ---
X = df.drop('left', axis=1)
y = df['left']
# Chia 80% dữ liệu để huấn luyện, 20% để kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Xây dựng và huấn luyện mô hình Random Forest ---
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- 4. Đánh giá hiệu suất và tính độ chính xác ---
# Dùng mô hình đã huấn luyện để dự đoán trên bộ dữ liệu kiểm tra (20%)
y_pred = model.predict(X_test)

# 4.1. TÍNH ĐỘ CHÍNH XÁC TỔNG THỂ (ACCURACY)
# Accuracy = (Tổng số lần dự đoán đúng) / (Tổng số dự đoán)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Độ Chính Xác Tổng Thể (Accuracy): {accuracy * 100:.2f}%")
print("-> Nghĩa là mô hình đã dự đoán đúng trạng thái (nghỉ/ở lại) cho khoảng 98-99% nhân viên trong tập dữ liệu kiểm tra.")

# 4.2. BÁO CÁO PHÂN LOẠI CHI TIẾT
# Cung cấp cái nhìn sâu hơn về hiệu suất của mô hình trên từng lớp (0: Ở lại, 1: Nghỉ việc)
print("\n📊 Báo Cáo Phân Loại Chi Tiết:")
print(classification_report(y_test, y_pred))

# 4.3. MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
# Cho thấy mô hình nhầm lẫn ở đâu
print("\n🤔 Ma Trận Nhầm Lẫn:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Vẽ ma trận nhầm lẫn cho dễ hình dung
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ở Lại', 'Nghỉ Việc'], yticklabels=['Ở Lại', 'Nghỉ Việc'])
plt.xlabel('Dự Đoán')
plt.ylabel('Thực Tế')
plt.title('Kết Quả Dự Đoán Của Mô Hình')
plt.show()


# --- 5. Lưu lại mô hình ---
joblib.dump(model, 'employee_churn_model.pkl')
print("\n💾 Đã lưu mô hình vào file 'employee_churn_model.pkl'")