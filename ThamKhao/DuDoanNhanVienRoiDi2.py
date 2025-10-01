import pandas as pd
import joblib

# --- 1. Tải lại mô hình đã được huấn luyện ---
try:
    model = joblib.load('employee_churn_model.pkl')
    print("✅ Tải mô hình thành công!")
except FileNotFoundError:
    print("❌ Lỗi: Không tìm thấy file 'employee_churn_model.pkl'.")
    print("Vui lòng chạy lại mã huấn luyện ở bước trước để tạo ra file này.")
    exit()

# --- 2. Lấy danh sách các cột mà mô hình đã học ---
# Đây là bước quan trọng để đảm bảo dữ liệu đầu vào có đúng định dạng
model_columns = ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'Department_RandD', 'Department_accounting',
       'Department_hr', 'Department_management', 'Department_marketing',
       'Department_product_mng', 'Department_sales', 'Department_support',
       'Department_technical', 'salary_low', 'salary_medium']

def du_doan_nhan_vien(satisfaction_level, last_evaluation, number_project,
                        average_montly_hours, time_spend_company, Work_accident,
                        promotion_last_5years, Department, salary):
    """
    Hàm này nhận thông tin của nhân viên và dự đoán khả năng rời đi.
    - Department phải là một trong các giá trị: 'sales', 'accounting', 'hr', 'technical', 'support', 'management', 'IT', 'product_mng', 'marketing', 'RandD'
    - salary phải là: 'low', 'medium', hoặc 'high'
    """
    
    # --- 3. Tạo một DataFrame rỗng với đúng các cột của mô hình ---
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0 # Khởi tạo dòng đầu tiên với tất cả giá trị là 0

    # --- 4. Điền thông tin cơ bản của nhân viên ---
    input_data['satisfaction_level'] = satisfaction_level
    input_data['last_evaluation'] = last_evaluation
    input_data['number_project'] = number_project
    input_data['average_montly_hours'] = average_montly_hours
    input_data['time_spend_company'] = time_spend_company
    input_data['Work_accident'] = Work_accident
    input_data['promotion_last_5years'] = promotion_last_5years

    # --- 5. Xử lý thông tin về Phòng ban (Department) và Mức lương (salary) ---
    # Đặt giá trị '1' vào cột tương ứng
    department_column = 'Department_' + Department
    if department_column in input_data.columns:
        input_data[department_column] = 1
    
    salary_column = 'salary_' + salary
    if salary_column in input_data.columns:
        input_data[salary_column] = 1

    # --- 6. Thực hiện dự đoán ---
    # predict() -> trả về 0 (Ở lại) hoặc 1 (Nghỉ việc)
    prediction = model.predict(input_data)[0]
    
    # predict_proba() -> trả về [xác suất ở lại, xác suất nghỉ việc]
    probability = model.predict_proba(input_data)[0][1]

    # --- 7. In kết quả ---
    print("\n------------------ KẾT QUẢ DỰ ĐOÁN ------------------")
    if prediction == 1:
        print(f"🚨 DỰ ĐOÁN: Nhân viên này có khả năng sẽ RỜI ĐI.")
    else:
        print(f"👍 DỰ ĐOÁN: Nhân viên này sẽ Ở LẠI.")
    
    print(f"📊 Xác suất rời đi: {probability * 100:.2f}%")
    print("----------------------------------------------------")


# --------------------------------------------------------------------------
#                           CÁCH SỬ DỤNG
# --------------------------------------------------------------------------
# Bây giờ bạn có thể gọi hàm với các thông tin khác nhau để kiểm tra.

print("\n--- Ví dụ 1: Nhân viên có nguy cơ nghỉ việc cao ---")
du_doan_nhan_vien(
    satisfaction_level=0.7,      # Mức độ hài lòng rất thấp
    last_evaluation=0.8,
    number_project=6,            # Làm nhiều dự án
    average_montly_hours=290,    # Thời gian làm việc rất cao
    time_spend_company=4,
    Work_accident=0,
    promotion_last_5years=0,
    Department='sales',
    salary='low'
)

print("\n--- Ví dụ 2: Nhân viên an toàn, có khả năng ở lại ---")
du_doan_nhan_vien(
    satisfaction_level=0.50,     
    last_evaluation=0.7,
    number_project=3,
    average_montly_hours=260,  
    time_spend_company=10,
    Work_accident=0,
    promotion_last_5years=1,
    Department='technical',
    salary='low'
)