import pickle
import joblib
import json
import numpy as np
import pandas as pd

__salary = None
__dept = None
__columns = None
__model = None

def get_predict_churn_employee(satisfaction_level, last_evaluation, number_project,
       average_montly_hours, time_spend_company, Work_accident,
       promotion_last_5years, Department, salary):
    
    try:
        loc_index = __dept.index(Department.lower())
    except:
        loc_index = -1
    
    try:
        loc_index1 = __salary.index(salary.lower())
    except:
        loc_index1 = -1

    if loc_index==0:
        Department = 0
    elif loc_index==1:
        Department = 1
    elif loc_index==2:
        Department = 2
    elif loc_index==3:
        Department = 3
    elif loc_index==4:
        Department = 4
    elif loc_index==5:
        Department = 5
    elif loc_index==6:
        Department = 6
    elif loc_index==7:
        Department = 7
    elif loc_index==8:
        Department = 8
    elif loc_index==9:
        Department = 9
        
    if loc_index1==0:
        salary = 0
    elif loc_index1==1:
        salary = 1
    elif loc_index1==2:
        salary = 2
        
    # new_sample = [[0.41, 0.43, 3, 153, 3, 0, 1, 1, 1]]
    new_sample = [[satisfaction_level, last_evaluation, number_project,
        average_montly_hours, time_spend_company, Work_accident,
        promotion_last_5years, Department, salary]]
    x = pd.DataFrame(new_sample)
    x.columns = __columns
   
    predict = str(__model.predict(x)[0])
    if predict == "0":
        result = "No Churn"
    else:
        result = "Churn"
        
    return result


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global  __dept
    global __salary
    global __columns

    with open("./artifacts/department_value.json", "r") as f:
        __dept = json.load(f)['department_value']
    with open("./artifacts/salary_value.json", "r") as f:
        __salary = json.load(f)['salary_value']
    with open("./artifacts/salary_value.json", "r") as f:
        __columns = json.load(f)['columns']

    global __model
    if __model is None:
        with open('./artifacts/model_HR.sav', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")

def get_department_names():
    return __dept

def get_salary_level():
    return __salary

def get_columns():
    return __columns

# def get_data_columns():
#     return __data_columns

if __name__ == '__main__':
    load_saved_artifacts()
    # print(get_department_names())
    # print(get_salary_level())
    # print(get_columns())
    # print(get_predict_churn_employee(0.8, 0.8, 3, 200, 6, 1, 1, "accounting", "high"))
