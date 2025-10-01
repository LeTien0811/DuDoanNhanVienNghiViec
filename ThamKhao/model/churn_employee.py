import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold

from xgboost import XGBClassifier
from mlxtend.plotting import plot_confusion_matrix


# 1. Phương pháp tiếp cận khi xây dựng một ML model

# Top-Down method:

# P2P Comparison method:

# 2. Đánh giá, lựa chọn Base model
df_data = pd.read_csv('HR_comma_sep.csv')
df_data.head()
df_data.info()

# a, Phân chia dữ liệu
# separate input (features) and output (target)
y = df_data[['left']]
X = df_data.drop('left', axis=1)

# separate train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.25)
print(X_train.shape)
print(X_test.shape)

# b, Mã hóa các Categorical features
# encode categorical features
encoder = OrdinalEncoder()
encoder.fit(X_train[['salary', 'Department']])
X_train[['salary', 'Department']] = encoder.transform(X_train[['salary', 'Department']])
X_test[['salary', 'Department']] = encoder.transform(X_test[['salary', 'Department']])

# create a dictionary of models
model_dict = {
                'LogisticRegression': LogisticRegression(max_iter=1000), 
                'GaussianNB': GaussianNB(), 
                'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=1), 
                'DecisionTreeClassifier': DecisionTreeClassifier(min_samples_split=25),
                'SVM': svm.SVC(kernel='rbf',probability=False),
                'RandomForestClassifier': RandomForestClassifier(n_estimators = 10, min_samples_split=2, max_depth=30),
            }

model_scores = []
model_names = []
for name, model in model_dict.items():
    model.fit(X_train, np.ravel(y_train))
    prediction = model.predict(X_test)
    acc_score = accuracy_score(y_test, prediction)
    model_scores.append(acc_score)
    model_names.append(name)
    
    print('*'*10 + name + '*'*10 + ': {}'.format(acc_score))

#3. Tối ưu hóa model
# a, Remove outlier

# draw histogram of some features
df_data.hist()
plt.tight_layout()

df_data.value_counts('salary').to_frame()

df_data.value_counts('Department').to_frame()

# không cần áp dụng kỹ thuật Remove Outlier nào cho bộ dữ liệu này.

# b, Impute Missing values
# Ở phần đầu ta đã biết rằng không tồn tại Missing values, nên cũng không cần áp dụng kỹ thuật Impute Missing values.

# c, Balance data

# sử dụng phương pháp Class Weight. 
# Cách thực hiện rất đơn giản, khi khai báo model, 
# ta chỉ cần thêm tham số class_weight=‘balanced’.

# compare portion of each class 
df_data.value_counts('left').to_frame()

# d, Scale data
# sử dụng là Normalization và Standarlization. 
# Mình sẽ áp dụng kỹ thuật Standarlization trong bài này.

# 3.2 Huấn luyện models

# get categorical and numerical columns
numerical_ix = X.select_dtypes(include=['int64', 'float64']).columns
categorical_ix = X.select_dtypes(include=['object', 'bool']).columns

# define ColumnTransformer to perform transform data
trans_list = [('cat', OneHotEncoder(), categorical_ix), ('num', StandardScaler(), numerical_ix)]
col_trans = ColumnTransformer(transformers=trans_list)

# create a dictionary of models
model_dict = {
                'LogisticRegression': LogisticRegression(max_iter=500, class_weight='balanced'), 
                'GaussianNB': GaussianNB(), 
                'KNeighborsClassifier': KNeighborsClassifier(), 
                'DecisionTreeClassifier': DecisionTreeClassifier(class_weight='balanced'),
                'SVM': svm.SVC(class_weight='balanced'),
                'RandomForestClassifier': RandomForestClassifier(class_weight='balanced'),
            }

# evaluate models in dictionary using cross validation with RepeatedStratifiedKFold
model_scores = []
model_names = []
for name, model in model_dict.items():
    pipeline = Pipeline(steps=[('preparation', col_trans), ('model', model)])
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    scores = cross_val_score(pipeline, X, np.ravel(y), scoring='accuracy', cv=cv, n_jobs=-1)
    model_scores.append(scores)
    model_names.append(name)
    print('*'*10 + name + '*'*10)
    print(np.mean(scores))
    print(np.std(scores))
    
# display evaluation result on violin chart
# fig = go.Figure()
# for model, score in zip(model_names, model_scores):
#     fig.add_trace(go.Violin(
#                             y=score,
#                             name=model,
#                             box_visible=True,
#                             meanline_visible=True)
#                         )
# fig.update_layout(
#     autosize=False,
#     width=1000,
#     height=800,
#     margin=dict(l=10, r=10, b=10, t=10),
#     # paper_bgcolor="LightSteelBlue",
# )
# fig.show()
# 3.3 Tune hyper-parameters cho RandomForestClassifier model
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
pipeline = Pipeline(steps=[('preparation', col_trans), ('rf', rf_model)])
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)

param_grid = {'rf__n_estimators': np.arange(50, 150, 10),
               'rf__max_features': ['auto', 'sqrt', 'log2'],
               'rf__max_depth': np.arange(10, 100, 10),
               'rf__min_samples_split': [2, 5, 10],
               'rf__min_samples_leaf': [1, 2, 4],
               'rf__criterion' :['gini', 'entropy'],
               'rf__bootstrap': [True, False]}

grid_pipeline = GridSearchCV(pipeline, param_grid, scoring= 'accuracy', n_jobs=-1, cv=cv)
results = grid_pipeline.fit(X, np.ravel(y))
print( ' Best Mean Accuracy: %.3f ' % results.best_score_)
print( ' Best Config: %s ' % results.best_params_)
# result 
# bestMeanAccuracy = 0.992 
# bestConfig = {'rf__bootstrap': False, 'rf__criterion': 'entropy', 'rf__max_depth': 30, 'rf__max_features': 'auto', 'rf__min_samples_leaf': 1, 'rf__min_samples_split': 2, 'rf__n_estimators': 70} 

# 4. Sử dụng model để dự đoán - Use the best model to make prediction on test set

# actually, test set is used to train model. So, this is not really valuable.
y_pred = grid_pipeline.predict(X_test)
# create confusion matrix
cm = confusion_matrix(y_test, y_pred)
# display classification report
print(classification_report(y_test, y_pred))

# display confusion matrix
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()


# 4.2 Dự đoán trên mẫu dữ liệu mới

# create new sample
new_sample = [[0.41, 0.43, 3, 153, 3, 1, 1, 'sales', 'medium']]
df_new_sample = pd.DataFrame(new_sample)
df_new_sample.columns = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'Department', 'salary']
# get only predicted class
class_pred = pipeline.predict(df_new_sample)[0]
print('Class = {}'.format(class_pred))

# get only predicted class
class_pred = grid_pipeline.predict(df_new_sample)[0]
print('Class = {}'.format(class_pred))

# get predicted class and it's probably coresponding
pd.DataFrame(grid_pipeline.predict_proba(df_new_sample)*100, columns=grid_pipeline.classes_)

# 5. Save và Load model
# way 1
pickle.dump(grid_pipeline, open('model.pkl', 'wb'))
# pickle.dump(grid_pipeline.best_estimator_, open('model.pkl', 'wb'))

# way 2, same result as way 1
joblib.dump(grid_pipeline, 'model.pkl')
# joblib.dump(grid_pipeline.best_estimator_, 'model.pkl')

# way 1
model = pickle.load(open('model.pkl', 'rb'))
# way 2, same result as way 1
model = joblib.load('model.pkl')

# use loaded model to make prediction
class_id = model.predict(df_new_sample)[0]
print(class_id)


# ===============================================================================

# Try to train model with XGBoost algorithm
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# encoder = OrdinalEncoder()
# encoder.fit(X_train[categorical_ix])
# X_train[categorical_ix] = encoder.transform(X_train[categorical_ix])
# X_test[categorical_ix] = encoder.transform(X_test[categorical_ix])

# lb_encoder = LabelEncoder()
# lb_encoder.fit(y_train)
# y_train = lb_encoder.transform(y_train)
# y_test = lb_encoder.transform(y_test)

model = XGBClassifier(use_label_encoder=False, verbosity=0, class_weight='balanced')
pipeline = Pipeline(steps=[('trans', col_trans), ('model', model)])

pipeline.fit(X_train, np.ravel(y_train))
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print('Accuracy: %.2f%%' % (accuracy*100))

# 4. Sử dụng model để dự đoán - Use the best model to make prediction on test set

# actually, test set is used to train model. So, this is not really valuable.
y_pred = pipeline.predict(X_test)
# create confusion matrix
cm = confusion_matrix(y_test, y_pred)
# display classification report
print(classification_report(y_test, y_pred))

# display confusion matrix
fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()


# 4.2 Dự đoán trên mẫu dữ liệu mới
# create new sample
new_sample = [[0.41, 0.43, 3, 153, 3, 1, 1, 'sales', 'medium']]
df_new_sample = pd.DataFrame(new_sample)
df_new_sample.columns = ['satisfaction_level', 'last_evaluation', 'number_project', 'average_montly_hours', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 'Department', 'salary']
# get only predicted class
class_pred = pipeline.predict(df_new_sample)[0]
print('Class = {}'.format(class_pred))

# get only predicted class
class_pred = pipeline.predict(df_new_sample)[0]
print('Class = {}'.format(class_pred))

# get predicted class and it's probably coresponding
pd.DataFrame(pipeline.predict_proba(df_new_sample)*100, columns=pipeline.classes_)
# Xác suất của class 0 là 99.91%, còn của class 1 là 0.09%.

# 5. Save và Load model
# way 1
pickle.dump(pipeline, open('model.pkl', 'wb'))
# pickle.dump(pipeline.best_estimator_, open('model.pkl', 'wb'))

# way 2, same result as way 1
joblib.dump(pipeline, 'model.pkl')
# joblib.dump(pipeline.best_estimator_, 'model.pkl')

# way 1
model = pickle.load(open('model.pkl', 'rb'))
# way 2, same result as way 1
model = joblib.load('model.pkl')

# use loaded model to make prediction
class_id = model.predict(df_new_sample)[0]
print(class_id)
