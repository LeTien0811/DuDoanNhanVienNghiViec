import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

class TrainModelDuDoanNhanVienServices:
    def __init__(self, df):
        try:
            # doc du lieu tu file
            self.df = pd.read_csv('HR_comma_sep.csv')
            # danh dau cac cot quan trong
            self.df = pd.get_dummies(self.df, columns=['Department', 'salary'], drop_first=True)
        except FileNotFoundError:
            return False
            
        
    def TrainModel(self):
        try:
            # chuan bi du leu cho mo hinh
            x  = self.df.drop('left', axis=1)
            y = self.df['left']
            x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

            # Xay dung va huan luyen mo hinh
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(x_train, y_train)

            # luu lai mo hinh da train
            joblib.dump(model, 'employee_churn_model.pkl')
            return True
        except:
            return False
