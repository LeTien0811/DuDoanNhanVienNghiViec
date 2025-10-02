import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import io


class TrainModelDuDoanNhanVienServices():
    def __init__(self, file):
        self.error = None
        try:
            fileData = file.read().decode('utf-8')
            # doc du lieu tu file
            self.df = pd.read_csv(io.StringIO(fileData))
            # danh dau cac cot quan trong
            self.df = pd.get_dummies(self.df, columns=['Department', 'salary'], drop_first=True)
        except Exception as e:
            self.error = f"Co Loi Xay Ra: {e}"

    def ShowError(self):
        return self.error

    def TrainModel(self):
        if not hasattr(self, "df"):
            self.error = "Khong Co Du Lieu De Train"
            return False
        try:
            # chuan bi du leu cho mo hinh
            #chia du lieu
            # Xác định biến mục tiêu (y) và các biến dự báo (X)
            x  = self.df.drop('left', axis=1)
            y = self.df['left']

            x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

            ResultData= []

            # Xay dung va huan luyen mo hinh
            RandomForest = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            RandomForest.fit(x_train, y_train)
            RandomForest_preds = RandomForest.predict(x_test)

            RandomForest_data = {
                "RandomForest": {
                "DoChinhXac": round(accuracy_score(y_test, RandomForest_preds)*100,2),
                "BaoCaoPhanLoai": classification_report(y_test, RandomForest_preds)
                },
            }
            ResultData.append(RandomForest_data)

            knn_model = KNeighborsClassifier()
            knn_model.fit(x_train, y_train)
            knn_preds = knn_model.predict(x_test)
            knn_data = {"Knn": {
                "DoChinhXac": round(accuracy_score(y_test, knn_preds)*100,2),
                "BaoCaoPhanLoai": classification_report(y_test, knn_preds) 
            },}
            ResultData.append(knn_data)

            svc_model = SVC(random_state=42, probability=True)
            svc_model.fit(x_train, y_train)
            svc_preds = svc_model.predict(x_test)
            svc_data = {"Svc": {
                "DoChinhXac": round(accuracy_score(y_test, svc_preds)*100,2),
                "BaoCaoPhanLoai": classification_report(y_test, svc_preds) 
            },}
            ResultData.append(svc_data)

            # luu lai mo hinh da train
            joblib.dump(RandomForest, 'RandomForest_model.pkl')
            joblib.dump(knn_model, 'Knn_model.pkl')
            joblib.dump(svc_model, 'Svc_model.pkl')
            
            if not ResultData:
                return False
            
            return ResultData
        except Exception as e:
            self.error = f"Co Loi Xay Ra: {e}"
            return False
