from django.urls import path
from . import views

urlpatterns = [
    path('', views.post_list, name='post_list'),
    path('1/', views.post_list_json, name='post_list'),
    path('2/', views.post_list_json2, name='post_list'),
]
