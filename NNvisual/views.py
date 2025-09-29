from django.shortcuts import render
from django.http import JsonResponse
from NNvisual.pytorch import TrainModel
from .config import NN_config
import threading
import asyncio
import time

def home(request):
    if not request.session.session_key:
        request.session.save()
    return render(request, 'NNvisual/Main.html')

def blog(request):
    return render(request,"NNvisual/Blog.html")

def graphs(request):
    return render(request, 'NNvisual/Graphs.html')

# def pytorch(request):
#     def run_training():
#         # try:
#         if request.session.session_key[:5] in NN_config.keys():
#             nn_config = NN_config[request.session.session_key[:5]]
#         else:
#             nn_config = NN_config["User"]

#         def run_dummy():
#             dummy_tm = TrainModel(
#                 epoch=nn_config["epoch"],
#                 lr=0.001,
#                 activation="relu",
#                 num=1,
#                 batch_size=16,
#                 session_id="dummy"
#             )
#             asyncio.run(dummy_tm.train())
            
#         dummy_thread = threading.Thread(target=run_dummy, daemon=True)
#         dummy_thread.start()
    
#         # Small delay to let dummy start
#         time.sleep(0.1)

        
#         tm = TrainModel(
#             nn_config["epoch"],
#             nn_config["learning_rate"],
#             nn_config["activation_function"],
#             nn_config["dataset"],
#             nn_config["batch_size"],
#             request.session.session_key[:5]
#                         )
        
#         asyncio.run(tm.train())
#         print("Training finished successfully")

#     return JsonResponse({"status": "Training started"})

import time

# Global dummy thread
_dummy_running = False
_dummy_thread = None

def ensure_dummy_running():
    """Ensure dummy training is running in background"""
    global _dummy_running, _dummy_thread
    
    if _dummy_running and _dummy_thread and _dummy_thread.is_alive():
        return  # Already running
    
    def run_dummy_continuous():
        global _dummy_running
        _dummy_running = True
        
        while _dummy_running:
            try:
                dummy_tm = TrainModel(
                    epoch=100,
                    lr=0.001,
                    activation="relu",
                    num=1,
                    batch_size=16,
                    session_id="dummy_global"
                )
                asyncio.run(dummy_tm.train())
                time.sleep(0.5)
            except Exception as e:
                print(f"Dummy training error: {e}")
                time.sleep(1)
    
    _dummy_thread = threading.Thread(target=run_dummy_continuous, daemon=True)
    _dummy_thread.start()
    print("Global dummy training started")

def pytorch(request):
    # Ensure dummy is running globally
    ensure_dummy_running()
    
    def run_training():
        try:
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
        except Exception as e:
            print("Error during training:", e)
    
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()
    
    return JsonResponse({"status": "Training started"})