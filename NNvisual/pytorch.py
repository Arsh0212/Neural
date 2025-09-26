import torch 
import time as time
from torch import nn 
import torch.nn.functional as func 
from asgiref.sync import async_to_sync
from sklearn.datasets import make_moons
from channels.layers import get_channel_layer


values,labels  = make_moons(n_samples=300,noise=0.2,random_state=42)
values = torch.FloatTensor(values)
labels = torch.FloatTensor(labels).unsqueeze(1)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2,8)
        self.fc2 = nn.Linear(8,8)
        self.output = nn.Linear(8,1)

    def forward(self,input,epoch):
        self.data = []
        first_nodes = func.relu(self.fc1(input))
        second_nodes = func.relu(self.fc2(first_nodes))
        output = self.output(second_nodes)
        

        if epoch % 10 == 0:
            # Helper function to round tensor to 2 decimals and convert to list
            def round_tensor(t):
                return [round(v, 2) for v in t.detach().tolist()]
            
            self.data.append([round_tensor(input.mean(dim=0))])
            self.data.append([round_tensor(first_nodes.mean(dim=0))])
            self.data.append([round_tensor(second_nodes.mean(dim=0))])
            self.data.append([round_tensor(output.mean(dim=0))])
            
            return output, self.data
        
        return output,None
    
class TrainModel:
    def __init__(self, epoch=200, lr = 0.01):
        torch.manual_seed(41)
        self.model = NeuralNetwork()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimized = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.epoch = epoch
        self.losses = []

    def send_web_data(self,message):
        try:
            self.channel_layer = get_channel_layer()
            if self.channel_layer:
                async_to_sync(self.channel_layer.group_send)(
                        message["group_name"], message
                    )
        except Exception as e:
            print("Error occured",e)

    def train(self):
        for i in range(self.epoch):
            train_start= time.time()
            predictions,data = self.model.forward(values,i)
            loss = self.criterion(predictions,labels)
            self.losses.append(loss.detach().numpy())

            if i%10 == 0 and data:
                pred = torch.sigmoid(predictions) > 0.5
                # self.graph_message(i,values,pred.tolist(),labels) 
                weights = []
                biases = []
                weights.append([[0]])
                biases.append([0])
                print(f"Epoch : {i} and loss : {loss}")

                for name, param in self.model.named_parameters():
                    # Detach tensor and convert to nested list
                    param_list = param.detach().tolist()
                    
                    # Round each element to 2 decimals
                    if "weight" in name:
                        weights.append([[round(v, 2) for v in row] for row in param_list])
                    elif "bias" in name:
                        biases.append([round(v, 2) for v in param_list])
                print("Phase 0:",time.time()-train_start)
                message = self.create_message(i,weights,biases,data,loss,1)
                self.send_web_data(message)
            print("Phase 1:",time.time() - train_start) # print something
            self.optimized.zero_grad()
            loss.backward()
            self.optimized.step()
            print("End:",time.time() - train_start)

    def create_message(self,epoch,weights,biases,nodes,loss,accuracy=1):
        message_data = {
                "epoch": epoch,
                "weights": weights,
                "biases": biases,
                "activated_nodes": nodes,
                "loss": float(loss.detach()),
                "accuracy": float(accuracy),
            }
            
        return {
                "type": "send_epoch_update",
                "group_name": "ws_train_main",
                "data": message_data
            }
    
    def graph_message(self,epoch,values,pred,labels):
        message = {
                "type": "training_update", 
                "group_name": "ws_train_graph",
                "data": {
                    "epoch": epoch,
                    "predicted": [p[0] for p in pred],
                    "x": values[: , 0].tolist(),
                    "y": values[: , 1].tolist(),
                    "labels": [e[0] for e in labels.tolist()]
                }
            }
        self.send_web_data(message)
