from django.shortcuts import render
from django.http import JsonResponse
import threading
import asyncio
from NNvisual.pytorch import TrainModel
from .config import NN_config


def home(request):
    if not request.session.session_key:
        request.session.save()
    return render(request, 'NNvisual/Main.html')

def blog(request):
    return render(request,"NNvisual/Blog.html")

def graphs(request):
    return render(request, 'NNvisual/Graphs.html')

def pytorch(request):
    def run_training():
        # try:
        if request.session.session_key[:5] in NN_config.keys():
            nn_config = NN_config[request.session.session_key[:5]]
        else:
            nn_config = NN_config["User"]
        tm = TrainModel(
            nn_config["epoch"],
            nn_config["learning_rate"],
            nn_config["activation_function"],
            nn_config["dataset"],
            nn_config["batch_size"],
            request.session.session_key[:5]
                        )
        
        asyncio.run(tm.train())
        print("Training finished successfully")
        # except Exception as e:
        #     print("Error during training:", e)

    # Run training in background thread
    thread = threading.Thread(target=run_training)
    thread.start()

    return JsonResponse({"status": "Training started"})