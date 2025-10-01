import pandas as pd
from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix


df_data = pd.read_csv('C:/Users/huy.nguyenq5/Downloads/projects_python/course_da/HR_Comma_Project/HR_comma_sep.csv')

df_data.info()

# =====Build model Machine Learning=====
# Step 1: Tách biến độc lập và biến phụ thuộc (biến mục tiêu ('left'))
X = df_data.drop('left', axis=1)
y = df_data[['left']]

#Step 2: Chia dữ liệu thành tập train và tập test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

encoder = OrdinalEncoder()

encoder.fit(X_train[['salary', 'Department']])

X_train[['salary', 'Department']] = encoder.transform(X_train[['salary', 'Department']])
X_test[['salary', 'Department']] = encoder.transform(X_test[['salary', 'Department']])


model = XGBClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
confusion = confusion_matrix(y_test, predictions)
y_test.value_counts()
fig, ax = plot_confusion_matrix(conf_mat=confusion,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)

print(classification_report(y_test, predictions))

new_sample = [[0.9, 0.9, 3, 153, 5, 0, 1, 2, 2]]

df_newSample = pd.DataFrame(new_sample)
df_newSample.columns = X.columns
# a = list(X.Department.unique())
pred = model.predict(df_newSample)[0]
print(pred)

import pickle
import joblib
pickle.dump(model, open('model_HR.pkl', 'wb'))
joblib.dump(model, open('model_HR.sav', 'wb'))

import json 
dept = {'department_value': list(X.Department.unique())}
with open ('department_value.json', 'w') as f:
   f.write(json.dumps(dept))
   
salary = {'salary_value': list(X.salary.unique()), 'columns':list(X.columns)}
with open ('salary_value.json', 'w') as f:
   f.write(json.dumps(salary))
# Explain:
#     1. Đánh giá các models: LogisticRegression, KNN, DecisionTreeClassifier,
#     SVM, RandomForestClassifier => accuracy => the best model for dataset 
#     2. Precision, Recall, F1-Score, Tune hyper-parameters
#     3. Crawl Data from Youtube
#     4. Build UI web local for Predict Price House (buoi 10)
#     5. Loyalty Database: connect DB, Collection Data, Aggregate Data, Build model, Model API (Swagger)
#     6. MLOps - cmd run/Prefect
#     7. Clustering (KMeans/DBSCAN/Hierachical) - Customers Segmentation


