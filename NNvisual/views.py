from django.shortcuts import render
from django.http import JsonResponse
from django.core.management import call_command
import threading


def home(request):
    return render(request, 'NNvisual/Main.html')

def blog(request):
    return render(request,"NNvisual/Blog.html")

def graphs(request):
    return render(request, 'NNvisual/Graphs.html')

def train(request):
    def run_training():
        call_command('train_model')  # runs your Command.handle()

    # Run training in background thread
    thread = threading.Thread(target=run_training)
    thread.start()

    return JsonResponse({"status": "Training started"})
