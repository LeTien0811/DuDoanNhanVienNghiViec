from flask import Flask, request, jsonify
import util

app = Flask(__name__)

@app.route('/get_dept_salary', methods=['GET'])
def get_location_names():
    response = jsonify({
        'Department': util.get_department_names(), 
        'salary': util.get_salary_level(),
        'columns': util.get_columns()
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/predict_churn_employee', methods=['GET', 'POST'])
def predict_home_price():
    satisfaction_level = float(request.form['satisfaction_level'])
    last_evaluation = float(request.form['last_evaluation'])
    number_project = float(request.form['number_project'])
    average_montly_hours = float(request.form['average_montly_hours'])
    time_spend_company = float(request.form['time_spend_company'])
    Work_accident = float(request.form['Work_accident'])
    promotion_last_5years = float(request.form['promotion_last_5years'])
    Department = request.form['Department']
    salary = request.form['salary']

    response = jsonify({
        'predict_left': util.get_predict_churn_employee(satisfaction_level, last_evaluation, number_project,average_montly_hours, time_spend_company, Work_accident,promotion_last_5years, Department, salary)
    })
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Prediction...")
    util.load_saved_artifacts()
    app.run()