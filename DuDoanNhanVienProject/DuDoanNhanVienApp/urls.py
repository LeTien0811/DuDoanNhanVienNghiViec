from django.urls import path 
from . import views

urlpatterns = [
    path('', views.indexs, name="index_View")
]
