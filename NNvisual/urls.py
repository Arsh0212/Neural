from django.urls import path
from . import views
urlpatterns = [
    path('',views.home,name="home"),
    path('NN/',views.home,name='home'),
    path('graphs/',views.graphs,name='graphs'),
    path('train/',views.pytorch,name='train'),
    path('blog/',views.blog,name="blog"),
    path('pytorch/',views.pytorch,name="pytorch"),
]