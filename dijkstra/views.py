from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from django.core import serializers
import numpy as np
import dijkstar

def index(request):
    src = request.GET.get('src', '123')
    dest = request.GET.get('dest', '234')
    graph = dijkstar.Graph.load("/Users/Ryan/Desktop/tensorly/graph.txt")
    cost_func = lambda u, v, e, prev_e: e['cost']
    try:
        ans = dijkstar.find_path(graph, src, dest, cost_func=cost_func)[0]
    except:
        print("No Path between these two vertex")
        ans = "No Path between these two vertex"
    response = {
        'trajectory': ans
    }
    return JsonResponse(response)