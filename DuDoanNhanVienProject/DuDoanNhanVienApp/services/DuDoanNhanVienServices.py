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
            input_data = pd.DataFrame(columns=self.model_columns)
            input_data.loc[0] = 0

            # dien thong tin vao
            input_data['satisfaction_level'] = satisfaction_level
            input_data['last_evaluation'] = last_evaluation
            input_data['number_project'] = number_project
            input_data['average_montly_hours'] = average_montly_hours
            input_data['time_spend_company'] = time_spend_company
            input_data['Work_accident'] = Work_accident
            input_data['promotion_last_5years'] = promotion_last_5years

            # Tien xu ly du lieu
            # xu ly thong tin ve phong ban va muc luong 
            department_column = 'Department_' + Department
            if department_column in input_data.columns:
                input_data.at[0,department_column] = 1
        
            salary_column = 'salary_' + salary
            if salary_column in input_data.columns:
                input_data.at[0,salary_column] = 1

            resultData = []

            # thuc hien du doan predict sẽ trả vê 0 hoac 1 o lại hoặc rời đ
            RandomForest_predict = self.RandomForest_model.predict(input_data)[0]
            # predict_probe trả ve xác suất
            RandomForest_predict_proba = self.RandomForest_model.predict_proba(input_data)[0]
            resultData.append( 
                {"Random_forest":{
                    "du_doan_roi_di": int(RandomForest_predict), 
                    "phan_tram_roi_di": float(RandomForest_predict_proba[1]),
                    "do_tin_cay_cua_du_doan": float(RandomForest_predict_proba[RandomForest_predict])
                }}
            )

            Knn_predict = self.Knn_model.predict(input_data)[0]
            Knn_predict_proba = self.Knn_model.predict_proba(input_data)[0]
            
            resultData.append( 
                {"Knn":{
                    "du_doan_roi_di": int(Knn_predict), 
                    "phan_tram_roi_di": float(Knn_predict_proba[1]),
                    "do_tin_cay_cua_du_doan": float(Knn_predict_proba[Knn_predict])
                }}
            )

            svc_prediction = self.Svc_model.predict(input_data)[0]
            try:
                svc_probabilities = self.Svc_model.predict_proba(input_data)[0]
                svc_phan_tram_roi_di = float(svc_probabilities[1])
                svc_do_tin_cay = float(svc_probabilities[svc_prediction])
            except AttributeError:
            # Nếu không có predict_proba, ta không thể cung cấp xác suất
                svc_phan_tram_roi_di = None 
                svc_do_tin_cay = 1.0 # Có thể mặc định là 1.0 vì predict không có xác suất đi kèm

            resultData.append({
                "Svc": {
                "du_doan_roi_di": int(svc_prediction),
                "phan_tram_roi_di": svc_phan_tram_roi_di,
                "do_tin_cay_cua_du_doan": svc_do_tin_cay
            }
            })
            if not resultData:
                return False
            return resultData
        except Exception as e:
            self.error = f"Co loi xay ra: {e}"
            return False

