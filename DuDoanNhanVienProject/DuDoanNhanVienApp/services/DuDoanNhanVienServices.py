import pandas as pd
import joblib
from DuDoanNhanVienApp.services.TrainModelDuDoanNhanVienServices import TrainModelDuDoanNhanVienServices

class DuDoanNhanVienServices:
    
    def __init__(self, model, model_columns):
        try:
            self.model = joblib.load('employee_churn_model.pkl')
            self.model_columns = ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'Department_RandD', 'Department_accounting',
       'Department_hr', 'Department_management', 'Department_marketing',
       'Department_product_mng', 'Department_sales', 'Department_support',
       'Department_technical', 'salary_low', 'salary_medium']
        except:
            train = TrainModelDuDoanNhanVienServices()
            check = train.TrainModel()
            if not check:
                return False
        
    def DuDoanNhanVien(self, satisfaction_level, last_evaluation, number_project,
                        average_montly_hours, time_spend_company, Work_accident,
                        promotion_last_5years, Department, salary):
        # tao data frame rong voi cac cot cua mo hinh
        input_data = pd.DataFrame(self.model_columns)
        input_data.loc[0] = 0

        # dien thong tin vao
        input_data['satisfaction_level'] = satisfaction_level
        input_data['last_evaluation'] = last_evaluation
        input_data['number_project'] = number_project
        input_data['average_montly_hours'] = average_montly_hours
        input_data['time_spend_company'] = time_spend_company
        input_data['Work_accident'] = Work_accident
        input_data['promotion_last_5years'] = promotion_last_5years
        input_data['Department'] = Department
        input_data['salary'] = salary

        # Tien xu ly du lieu
        # xu ly thong tin ve phong ban va muc luong 
        department_column = 'Department_' + Department
        if department_column in input_data.columns:
            input_data[department_column] = 1
        
        salary_column = 'salary_' + salary
        if salary_column in input_data.columns:
            input_data[salary_column] = 1
        
        # thuc hien du doan predict sẽ trả vê 0 hoac 1 o lại hoặc rời đi
        prediction = self.model.predict(input_data)[0]

        # predict_probe trả ve xác suất
        probability = self.model.predict_probe(input_data)[0][1]

        