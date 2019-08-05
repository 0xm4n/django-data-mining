from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

def ann(request):
    return HttpResponse('ann')

def lstm(request):
    return HttpResponse('lstm')