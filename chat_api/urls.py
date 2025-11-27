from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat_message, name='chat_message'),
    path('history/', views.chat_history, name='chat_history'),
]