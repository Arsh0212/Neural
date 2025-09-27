from django.shortcuts import render
from django.http import JsonResponse
import threading
import asyncio
from NNvisual.pytorch import TrainModel
from .models import NeuralNetwork


def home(request):
    return render(request, 'NNvisual/Main.html')

def blog(request):
    return render(request,"NNvisual/Blog.html")

def graphs(request):
    return render(request, 'NNvisual/Graphs.html')

def pytorch(request):
    def run_training():
        try:
            db_data = NeuralNetwork.objects.get(id=1)
            tm = TrainModel(db_data.epoch,db_data.learning_rate,db_data.activation_function, db_data.dataset)
            asyncio.run(tm.train())
            print("Training finished successfully")
        except Exception as e:
            print("Error during training:", e)

    # Run training in background thread
    thread = threading.Thread(target=run_training)
    thread.start()

    return JsonResponse({"status": "Training started"})