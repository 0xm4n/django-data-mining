# Django Server for Data Mining
![](https://img.shields.io/travis/zhenyit/django-data-mining) 
![](https://img.shields.io/github/license/zhenyit/django-data-mining) 

This project uses Django to provide RESTful Web Services for data mining in your app. Created by Zhenyi Tang.

## Overview
django-data-mining is a powerful and flexible server which provides a simple yet highly extensible architecture. 

It is hard to add up some python based data mining functions to an exsiting back-end server(Node.js). Although Node.js provides a  "child_process" module to run a python script and uses "stdout" to listen for the output, it has a poor scalability and difficult to debug. This project sets up a python server to run the data mining functions and return the output to the back-end in a JSON form.
#### Architecture
![](https://upload-images.jianshu.io/upload_images/17071502-e584f6ab7d7e1471.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240 "Software achitecture")
## Features
+ Cluster
  + k-means
  + hierarchical clustering
+ Neural network
  + artificial neural networks
  + Long short-term memory
+ Recommendation
  + collaborative filtering 
  
## Quick Start
Let's pretend you want to add up some other functions to this server. 

First, create a new app:
```
$ python manage.py startapp APP_NAME
```
Second, append this app to _INSTALLED_APPS_ list in **mlserver/settings.py**
```python
INSTALLED_APPS = [
    # Append the APP_NAME at the end of this array
]
```
Now, create an "_urls.py_" file in the directory of your new app (  ) and include it to the project. Here is the urlpatterns of your new app.
```python
# File path: APP_NAME/urls.py
from django.urls import path
from . import views
#
urlpatterns = [
    path('RELETIVE_PATH', views.FUNCTION_NAME),    
]
```

```python
# File path: mlserver/urls.py
from django.contrib import admin
from django.urls import path,include
urlpatterns = [
    # include the above patterns here
    path('APP_NAME/', include('APP_NAME.urls')),
]
```
When you send a request to _"APP_NAME/RELETIVE_PATH"_ the server will call the FUNCTION_NAME now.
