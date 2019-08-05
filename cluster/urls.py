from django.urls import path
from . import views

urlpatterns = [
    path('kmeans', views.kmeans),    
]
