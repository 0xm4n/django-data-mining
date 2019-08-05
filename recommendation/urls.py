from django.urls import path
from . import views

urlpatterns = [
    path('dijkstra', views.dijkstra),    
]
