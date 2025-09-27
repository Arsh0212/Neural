import torch
import time
from torch import nn
import torch.nn.functional as func
from asgiref.sync import async_to_sync
from sklearn.datasets import make_moons
from channels.layers import get_channel_layer
import threading

values, labels = make_moons(n_samples=300, noise=0.2, random_state=42)
values = torch.FloatTensor(values)
labels = torch.FloatTensor(labels).unsqueeze(1)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, input, epoch):
        self.data = []
        first_nodes = func.relu(self.fc1(input))
        second_nodes = func.relu(self.fc2(first_nodes))
        output = self.output(second_nodes)

        if epoch % 10 == 0:
            # Round tensors to 2 decimals
            def round_tensor(t):
                return [round(v, 2) for v in t.detach().tolist()]

            self.data.append([round_tensor(input.mean(dim=0))])
            self.data.append([round_tensor(first_nodes.mean(dim=0))])
            self.data.append([round_tensor(second_nodes.mean(dim=0))])
            self.data.append([round_tensor(output.mean(dim=0))])

            return output, self.data

        return output, None


class TrainModel:
    def __init__(self, epoch=200, lr=0.01):
        torch.manual_seed(41)
        self.model = NeuralNetwork()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimized = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epoch = epoch
        self.losses = []

    # --- Threaded WebSocket sending ---
    def send_web_data_threaded(self, message):
        def worker(msg):
            try:
                channel_layer = get_channel_layer()
                if channel_layer:
                    async_to_sync(channel_layer.group_send)(
                        msg["group_name"], msg
                    )
            except Exception as e:
                print("Error in sending thread:", e)

        threading.Thread(target=worker, args=(message,)).start()

    # --- Training loop ---
    def train(self):
        for i in range(self.epoch):
            start_time = time.time()
            predictions, data = self.model.forward(values, i)
            loss = self.criterion(predictions, labels)
            self.losses.append(loss.detach().numpy())
            print(f"Epoch {i} Phase 1:",time.time()-start_time)

            if i % 10 == 0 and data:
                pred = torch.sigmoid(predictions) > 0.5

                # Prepare weights and biases once
                weights = [[[0]]]
                biases = [[0]]
                for name, param in self.model.named_parameters():
                    param_list = param.detach().tolist()
                    if "weight" in name:
                        weights.append([[round(v, 2) for v in row] for row in param_list])
                    elif "bias" in name:
                        biases.append([round(v, 2) for v in param_list])

                message = self.create_message(i, weights, biases, data, loss)
                self.send_web_data_threaded(message)

            self.optimized.zero_grad()
            loss.backward()
            self.optimized.step()

            # Optional: print epoch time
            if i % 10 == 0:
                print(f"Epoch {i}, loss: {loss.item():.4f}, time: {time.time()-start_time:.2f}s")

    # --- Message creation ---
    def create_message(self, epoch, weights, biases, nodes, loss, accuracy=1):
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
