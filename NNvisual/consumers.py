from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
from .config import NN_config
import json

class NeuralNetworkConsumer(AsyncWebsocketConsumer):
    async def connect(self):
  
        session = self.scope["session"]
        if not session.session_key:
            await sync_to_async(session.save)()

        self.session_id = session.session_key[:5]

        if "main" in self.scope["path"]:
            self.group_name = "ws_train_main_"+self.session_id
        elif "metrics" in self.scope["path"]:
            self.group_name = "ws_train_metrics_"+self.session_id
        elif "graph" in self.scope["path"]:
            self.group_name = "ws_train_graph_"+self.session_id
        else:
            self.group_name = "ws_train_default"

        
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        print(self.group_name)
        await self.accept()

        if self.session_id in NN_config.keys():
            nn_config = NN_config[self.session_id]
        else:
            nn_config = NN_config["User"]
            NN_config[self.session_id] = NN_config["User"]

        await self.send(text_data=json.dumps({
            "type": "config",
            "config": {
                "epochs": nn_config["epoch"],
                "batchSize": nn_config["batch_size"],
                "learningRate": nn_config["learning_rate"],
                "activationFunction": nn_config["activation_function"],
                "datasetFunction" : nn_config["dataset"]
            }
        }))

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data):
        # from .models import NeuralNetwork
        data = json.loads(text_data)
        # You could use this to pause/resume training, etc.
        if data.get("type") == "config": 
            data = data.get("config") 
            # print(data)

            NN_config[self.session_id] = {
                "epoch" : data.get("epochs"),
                "batch_size" : data.get("batchSize"),
                "learning_rate" : data.get("learningRate"),
                "activation_function" : data.get("activationFunction"),
                "dataset" : data.get("datasetFunction")
            }

    # Receive message from the group (from train_model.py)
    async def send_epoch_update(self, event):
        # event["data"] contains the payload sent from WSLogger callback
        if "ws_train_main" in self.group_name:
            await self.send(text_data=json.dumps(event))

    async def training_update(self, event):
        # event["data"] contains the payload sent from train_model.py
        if "ws_train_graph" in self.group_name:
            await self.send(text_data=json.dumps(event))
