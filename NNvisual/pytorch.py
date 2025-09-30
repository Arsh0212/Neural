import torch
import time
import threading
from torch import nn
import torch.nn.functional as func
from asgiref.sync import async_to_sync
from torch.utils.data import TensorDataset,DataLoader
from channels.layers import get_channel_layer
from sklearn.datasets import make_moons, make_blobs, make_circles, make_classification

def get_dataset(num: int):
    if num == 1:
        values, labels = make_moons(n_samples=200, noise=0.2, random_state=42)
    elif num == 2:
        values, labels = make_circles(n_samples=200, noise=0.2, random_state=42)
    elif num == 3:
        values, labels = make_blobs(
            n_samples=200,
            centers=2,
            n_features=2,
            cluster_std=1.5,
            random_state=42
        )
    elif num == 4:
        values, labels = make_classification(
            n_samples=200,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            n_classes=2,
            random_state=42
        )
    else:
        raise ValueError("Invalid dataset number (choose 1–4)")
    
    values = torch.FloatTensor(values)
    labels = torch.FloatTensor(labels).unsqueeze(1)
    return values, labels

ACTIVATIONS = {
    "relu": func.relu,
    "sigmoid": func.sigmoid,
    "tanh": func.tanh,
    "linear": lambda x: x
}

def get_activation(name: str):
    """Return activation function by name, defaults to identity if not found"""
    return ACTIVATIONS.get(name.lower(), lambda x: x)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, input, epoch, activation):
        self.data = []
        activation = get_activation(activation)
        first_nodes = activation(self.fc1(input))
        second_nodes = activation(self.fc2(first_nodes))
        output = self.output(second_nodes)

        if epoch % 2 == 0:
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
    def __init__(self, epoch, lr, activation, num, batch_size, session_id):
        torch.manual_seed(41)
        self.model = NeuralNetwork()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimized = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.epoch = epoch
        self.num = num
        self.batch_size = batch_size
        self.activation = activation
        self.session_id = session_id
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
    async def train(self):
        values, labels = get_dataset(self.num)
        dataset = TensorDataset(values, labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for i in range(self.epoch):
            start_time = time.time()

            for batch_values, batch_labels in loader:
                predictions, data = self.model.forward(batch_values, i, self.activation)
                loss = self.criterion(predictions, batch_labels)

                self.optimized.zero_grad()
                loss.backward()
                self.optimized.step()

            self.losses.append(loss.item())

            if i % 2 == 0 and data:
                with torch.no_grad():
                    # Get predictions for the ENTIRE dataset
                    full_predictions, _ = self.model.forward(values, i, self.activation)
                    pred_probs = torch.sigmoid(full_predictions)
                    pred = (pred_probs > 0.5).float()

                    # Prepare weights and biases for neural network visualization
                    weights = [[[0]]]
                    biases = [[0]]
                    for name, param in self.model.named_parameters():
                        param_list = param.detach().tolist()
                        if "weight" in name:
                            weights.append([[round(v, 2) for v in row] for row in param_list])
                        elif "bias" in name:
                            biases.append([round(v, 2) for v in param_list])

                    # Send neural network layer data
                    nn_message = self.create_message(i, weights, biases, data, loss)
                    self.send_web_data_threaded(nn_message)

                    # Send training data for graph visualization
                    graph_message = self.create_training_update_message(
                        i, values, labels, pred
                    )
                    self.send_web_data_threaded(graph_message)

            if i % max(1, self.epoch//10) == 0 and self.session_id != "dummy_global":
                print(f"Epoch {i}, loss: {loss.item():.4f}, time: {time.time()-start_time:.2f}s")

    # --- Message creation for neural network visualization ---
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
            "group_name": "ws_train_main_"+self.session_id,
            "data": message_data
        }

    # --- Message creation for graph visualization ---
    def create_training_update_message(self, epoch, values, labels, predictions):
        """Create training update message for graph visualization"""
        
        # Convert tensors to lists and flatten if needed
        x_data = values[:, 0].detach().tolist()
        y_data = values[:, 1].detach().tolist()
        labels_data = labels.squeeze().detach().tolist()
        pred_data = predictions.squeeze().detach().tolist()
        
        message_data = {
            "epoch": epoch,
            "x": x_data,
            "y": y_data,
            "labels": labels_data,
            "predicted": pred_data
        }
        
        return {
            "type": "training_update",
            "group_name": "ws_train_graph_"+self.session_id,
            "data": message_data
        }