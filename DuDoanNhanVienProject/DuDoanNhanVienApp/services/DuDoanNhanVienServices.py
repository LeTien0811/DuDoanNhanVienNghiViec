import pandas as pd
import joblib
from DuDoanNhanVienApp.services.TrainModelDuDoanNhanVienServices import TrainModelDuDoanNhanVienServices

class DuDoanNhanVienServices:
    
    def __init__(self):
        self.error = None
        try:
            self.RandomForest_model = joblib.load('RandomForest_model.pkl')
            self.Knn_model = joblib.load('Knn_model.pkl')
            self.Svc_model = joblib.load('Svc_model.pkl')

            self.model_columns = ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company', 'Work_accident',
       'promotion_last_5years', 'Department_RandD', 'Department_accounting',
       'Department_hr', 'Department_management', 'Department_marketing',
       'Department_product_mng', 'Department_sales', 'Department_support',
       'Department_technical', 'salary_low', 'salary_medium']
            
        except Exception as e:
            self.error = f"loi: {e}"
    
    def ShowError(self):
        return self.error
    
    def DuDoanNhanVienText(self, satisfaction_level, last_evaluation, number_project,
                        average_montly_hours, time_spend_company, Work_accident,
                        promotion_last_5years, Department, salary):
        try:
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

            resultData = []

            # thuc hien du doan predict sẽ trả vê 0 hoac 1 o lại hoặc rời đ
            RandomForest_KhaNangRoiDi = self.RandomForest_model.predict(input_data)[0]
            # predict_probe trả ve xác suất
            RandomForest_probability = self.RandomForest_model.predict_proba(input_data)[0][1]
            RandomForest_XacXuatPhanTram = round(RandomForest_probability * 100, 2)
            resultData.append( 
                {"Random_forest":{
                    "KhaNangRoiDi": RandomForest_KhaNangRoiDi, 
                    "XacXuatPhanTram": RandomForest_DuDoanNhanVienServicesXacXuatPhanTram
                }}
            )

            Knn_KhaNangRoiDi = self.Knn_model.predict(input_data)[0]
            Knn_probability = self.Knn_model.predict_proba(input_data)[0][1]
            Knn_XacXuatPhanTram = round(Knn_probability * 100, 2)
            resultData.append( 
                {"Knn":{
                    "KhaNangRoiDi": Knn_KhaNangRoiDi, 
                    "XacXuatPhanTram": Knn_XacXuatPhanTram
                }}
            )

            Svc_KhaNangRoiDi = self.Svc_model.predict(input_data)[0]
            Svc_probability = self.Svc_model.predict_proba(input_data)[0][1]
            Svc_XacXuatPhanTram = round(Svc_probability * 100, 2)
            resultData.append( 
                {"Svc":{
                    "KhaNangRoiDi": Svc_KhaNangRoiDi, 
                    "XacXuatPhanTram": Svc_XacXuatPhanTram
                }}
            )
            if not resultData:
                return False
            return resultData
        except Exception as e:
            self.error = f"Co loi xay ra: {e}"
            return False

