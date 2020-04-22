from django.urls import path
from . import views

urlpatterns = [
    path('1', views.post_list_json, name='post_list'),

]
