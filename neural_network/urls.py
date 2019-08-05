from django.urls import path
from . import views

urlpatterns = [
    path('ann', views.ann),
    path('lstm', views.lstm), 
]