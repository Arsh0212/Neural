from django.shortcuts import render
from django.http import JsonResponse
from django.core.management import call_command
import threading
from NNvisual.pytorch import TrainModel


def home(request):
    return render(request, 'NNvisual/Main.html')

def blog(request):
    return render(request,"NNvisual/Blog.html")

def graphs(request):
    return render(request, 'NNvisual/Graphs.html')

def train(request):
    def run_training():
        print("Running Command")
        call_command('train_model')  # runs your Command.handle()

    # Run training in background thread
    thread = threading.Thread(target=run_training)
    thread.start()

    return JsonResponse({"status": "Training started"})

def pytorch(request):
    def run_training():
        try:
            tm = TrainModel()
            tm.train()
            print("Training finished successfully")
        except Exception as e:
            print("Error during training:", e)

    # Run training in background thread
    thread = threading.Thread(target=run_training)
    thread.start()

    return JsonResponse({"status": "Training started"})