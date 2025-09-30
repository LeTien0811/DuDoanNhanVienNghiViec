function getWorkAccidentValue() {
  var uiWorkAccident = document.getElementsByName("uiWorkAccident");
  for(var i in uiWorkAccident) {
    if(uiWorkAccident[i].checked) {
        return parseInt(i)+1;
    }
  }
  return -1; // Invalid Value
}

function getpromotion_last_5yearsValue() {
  var uipromotion_last_5years = document.getElementsByName("uipromotion_last_5years");
  for(var i in uipromotion_last_5years) {
    if(uipromotion_last_5years[i].checked) {
        return parseInt(i)+1;
    }
  }
  return -1; // Invalid Value
}

function onClickedPredictChurnEmployee() {
  console.log("Predict churn employee button clicked");
  var satisfaction_level = document.getElementById("uisatisfaction_level");
  var last_evaluation = document.getElementById("uilast_evaluation");
  var number_project = document.getElementById("uinumber_project");
  var average_montly_hours = document.getElementById("uiaverage_montly_hours");
  var time_spend_company = document.getElementById("uitime_spend_company");  
  //var workAccident = getWorkAccidentValue();
  //var promotion_last_5years = getpromotion_last_5yearsValue();
  var Work_accident = document.getElementsByName("uiWork_accident");
  var promotion_last_5years = document.getElementsByName("uipromotion_last_5years");
  var Department = document.getElementById("uiDepartment");
  var salary = document.getElementById("uisalary");
  var uileft = document.getElementById("uileft");

  var url = "http://127.0.0.1:5000/predict_churn_employee"; 
  //var url = "/api/predict_home_price";
  $.post(url, {
      satisfaction_level: parseFloat(satisfaction_level.value),
      last_evaluation: parseFloat(last_evaluation.value),
      number_project: parseFloat(number_project.value),
      average_montly_hours: parseFloat(average_montly_hours.value),
      time_spend_company: parseFloat(time_spend_company.value),
      Work_accident: parseFloat(Work_accident.value),
      promotion_last_5years: parseFloat(promotion_last_5years.value),
      Department: Department.value,
      salary: salary.value
  },function(data, status) {
      console.log(data.predict_left);
      uileft.innerHTML = "<h2>" + data.predict_left.toString() + "</h2>";
      console.log(status);
  });
}

function onPageLoad() {
  console.log( "document loaded" );
  var url = "http://127.0.0.1:5000/get_dept_salary"; 
  //var url = "/api/get_location_names"; 
  $.get(url,function(data, status) {
      console.log("got response for get_department_names & get_salary_level request");
      if(data) {
          var Department = data.Department;
          var salary = data.salary;
          
          
          var uiDepartment = document.getElementById("uiDepartment");
          $('#uiDepartment').empty();
          for(var i in Department) {
              var opt1 = new Option(Department[i]);
              $('#uiDepartment').append(opt1);
          }
          var uisalary = document.getElementById("uisalary");
          $('#uisalary').empty();
          for(var i in salary) {
              var opt2 = new Option(salary[i]);
              $('#uisalary').append(opt2);
          }
      }
  });
}

window.onload = onPageLoad;
