from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader
from DuDoanNhanVienApp.services.TrainModelDuDoanNhanVienServices import TrainModelDuDoanNhanVienServices
from DuDoanNhanVienApp.services.DuDoanNhanVienServices import DuDoanNhanVienServices
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json

@csrf_exempt
# Create your views here.
def indexs(request):
    template = loader.get_template("Home.html");
    return HttpResponse(template.render())

@csrf_exempt
def Trainning(request):
    if request.method == "POST" and request.FILES.get("TrainDataFile"):
        try:
            file = request.FILES["TrainDataFile"]
            trainning = TrainModelDuDoanNhanVienServices(file)
            result = trainning.TrainModel()
            if result == False:
                error = trainning.ShowError()
                return JsonResponse({
                    "error": f"Khong Train Duoc: {error}"
                }, status=400)
            
            return JsonResponse({
                "message": "Train Thanh Cong",
                "data": result
            }, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    return JsonResponse({"error": "method error"}, status=400) 

@csrf_exempt
def DuDoanRoiDiText(request):
    if request.method == "POST":
        satisfaction_level = request.POST.get("satisfaction_level")
        last_evaluation = request.POST.get("last_evaluation")
        number_project = request.POST.get("number_project")
        average_montly_hours = request.POST.get("average_montly_hours")
        time_spend_company = request.POST.get("time_spend_company")
        Work_accident = request.POST.get("Work_accident")
        promotion_last_5years = request.POST.get("promotion_last_5years")
        Department = request.POST.get("Department")
        salary = request.POST.get("salary")
        fields = [satisfaction_level, last_evaluation, number_project, average_montly_hours,
                  time_spend_company, Work_accident, promotion_last_5years, Department, salary]
        if any(f is None or f == "" for f in fields):
            return JsonResponse({"error": "empty"}, status=400)

        try:
            DuDoanIng = DuDoanNhanVienServices()
            result = DuDoanIng.DuDoanNhanVienText(
                float(satisfaction_level),
                float(last_evaluation),
                int(number_project),
                int(average_montly_hours),
                int(time_spend_company),
                int(Work_accident),
                int(promotion_last_5years),
                Department,
                salary
            )
            if result == False:
                error = DuDoanIng.ShowError()
                return JsonResponse({
                    "error": f"Loi: {error}"
                }, status=400)
            return JsonResponse({
                "message": "Thanh Cong",
                "data": result
            }, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
    return JsonResponse({"error": "method error"}, status=400) 