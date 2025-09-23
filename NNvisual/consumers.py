from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async
import json

class NeuralNetworkConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        from .models import NeuralNetwork
        if "main" in self.scope["path"]:
            self.group_name = "ws_train_main"
        elif "metrics" in self.scope["path"]:
            self.group_name = "ws_train_metrics"
        elif "graph" in self.scope["path"]:
            self.group_name = "ws_train_graph"
        else:
            self.group_name = "ws_train_default"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        print(self.group_name)
        await self.accept()
        nn, created = await sync_to_async(NeuralNetwork.objects.get_or_create)(
            id=1,  # or any unique field to check existence
            defaults={
                'epoch': 100,
                'batch_size': 30,
                'learning_rate': 0.01,
                'activation_function':'tanh',
                'dataset':1
            }
        )
        await self.send(text_data=json.dumps({
            "type": "config",
            "config": {
                "epochs": nn.epoch,
                "batchSize": nn.batch_size,
                "learningRate": nn.learning_rate,
                "activationFunction": nn.activation_function,
                "datasetFunction" : nn.dataset
            }
        }))

    async def disconnect(self, close_code):
        channel_name = "neural_network_updates"
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data):
        data = json.loads(text_data)
        # You could use this to pause/resume training, etc.
        if data.get("type") == "config": 
            data = data.get("config") 
            print(data)          
            await sync_to_async(NeuralNetwork.objects.update_or_create)(
                id = 1,
                defaults= {
                "epoch" : data.get("epochs"),
                "batch_size" : data.get("batchSize"),
                "learning_rate" : data.get("learningRate"),
                "activation_function" : data.get("activationFunction"),
                "dataset" : data.get("datasetFunction")
                }
            )

    # Receive message from the group (from train_model.py)
    async def send_epoch_update(self, event):
        # event["data"] contains the payload sent from WSLogger callback
        await self.send(text_data=json.dumps(event))

    async def training_update(self, event):
        # event["data"] contains the payload sent from train_model.py
        if event["type"] == "training_update":
            await self.send(text_data=json.dumps(event))
