from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.http import JsonResponse
from django.core import serializers
from . import models
import numpy as np

from sklearn.cluster import KMeans

def kmeans(request):
    response = {
        'label': [],
        'cluster_centers': []
    }

    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])

    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

    response['label'] = kmeans.labels_.tolist()
    response['cluster_centers'] = kmeans.cluster_centers_.tolist()

    # response = serializers.serialize('python', response)

    return JsonResponse(response)