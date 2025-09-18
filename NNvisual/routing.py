from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/training/', consumers.NeuralNetworkConsumer.as_asgi()),
    # Add more websocket routes as needed
]