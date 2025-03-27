from django.urls import path
from . import views

app_name = 'main'

urlpatterns = [
    path('', views.home, name='home'),
    path('algorithm1/', views.algorithm1, name='about'),
    path('algorithm2/', views.algorithm2, name='services'),
    path('algorithm3/', views.algorithm3, name='blog'),
    path('algorithm4/', views.algorithm4, name='contact'),
    path('algorithm5/', views.algorithm5, name='dashboard'),
] 