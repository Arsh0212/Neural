from django.core.management.base import BaseCommand
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
import os, warnings
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress INFO & WARNING logs
warnings.filterwarnings("ignore")  # suppress Python warnings
import tensorflow as tf
import numpy as np
from NNvisual.training import build_model, get_dataset
import time
from NNvisual.models import NeuralNetwork
import threading
from queue import Queue, Empty
import json

# Pre-compute sigmoid for better performance
@tf.function
def sigmoid_tf(x):
    return tf.nn.sigmoid(x)

def sigmoid_numpy(x):
    # Clip to prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

class Command(BaseCommand):
    help = "Train the neural network and broadcast updates"

    def __init__(self):
        super().__init__()
        self.channel_layer = None
        self.websocket_queue = Queue(maxsize=50)  # Limit queue size
        self.websocket_thread = None
        self.stop_event = threading.Event()
        self.last_update_time = 0
        self.min_update_interval = 0.1  # Minimum 500ms between updates

    def setup_websocket_handler(self):
        """Setup async websocket message handler in separate thread"""
        def websocket_worker():
            while not self.stop_event.is_set():
                try:
                    message = self.websocket_queue.get(timeout=0.1)
                    if message is None:  # Shutdown signal
                        break
                    
                    group_name = message.get("group_name")
                    
                    # Compress data if too large
                    if "data" in message and len(json.dumps(message["data"])) > 10000:
                        message["data"] = self.compress_data(message["data"])
                    
                    async_to_sync(self.channel_layer.group_send)(
                        group_name, message
                    )
                    self.websocket_queue.task_done()
                except Empty:
                    continue
                except Exception as e:
                    print(f"WebSocket error: {e}")
                    continue
        
        self.websocket_thread = threading.Thread(target=websocket_worker, daemon=True)
        self.websocket_thread.start()

    def compress_data(self, data):
        """Compress large data payloads"""
        compressed = {}
        for key, value in data.items():
            if key == "weights" and isinstance(value, list):
                # Only send first and last layer weights, sample middle layers
                if len(value) > 3:
                    compressed[key] = [value[0], value[-1]]  # First and last only
                else:
                    compressed[key] = value
            elif key == "activated_nodes" and isinstance(value, list):
                # Sample nodes if too many
                if len(value) > 0 and isinstance(value[0], list) and len(value[0]) > 100:
                    compressed[key] = [arr[:50] for arr in value]  # First 50 nodes only
                else:
                    compressed[key] = value
            else:
                compressed[key] = value
        return compressed

    def queue_websocket_message(self, message):
        """Queue websocket message for async sending with rate limiting"""
        current_time = time.time()
        
        # Rate limiting
        if current_time - self.last_update_time < self.min_update_interval:
            return  
        # Skip this update
            
        try:
            # Non-blocking put - if queue is full, drop the message
            self.websocket_queue.put_nowait(message)
            self.last_update_time = current_time
        except:
            # Queue full or other error, skip this update
            pass

    def handle(self, *args, **kwargs):
        print("Train Command called")
        # Get dataset once
        feature_train, label_train, feature_test, label_test, train_first, name = get_dataset()
        
        # Convert to TensorFlow tensors for better performance
        feature_train_tf = tf.constant(feature_train, dtype=tf.float32)
        label_train_tf = tf.constant(label_train, dtype=tf.float32)
        feature_test_tf = tf.constant(feature_test, dtype=tf.float32)
        label_test_tf = tf.constant(label_test, dtype=tf.float32)
        train_first_tf = tf.constant(train_first, dtype=tf.float32)
        
        print("Build started", start := time.time())
        model = build_model()
        print("Build ended", time.time()-start)
        
        # Setup WebSocket handling
        self.channel_layer = get_channel_layer()
        self.setup_websocket_handler()

        # Pre-compile prediction function
        @tf.function
        def fast_predict(x):
            return model(x, training=False)

        class OptimizedWSLogger(tf.keras.callbacks.Callback):
            def __init__(self, parent_command, train_sample, update_epoch_count=0):
                super().__init__()
                self.parent = parent_command
                self.train_sample = train_sample
                self.update_epoch_count = update_epoch_count
                self.last_detailed_update = 0
                self.activation_functions = {
                    'relu': lambda x: tf.nn.relu(x),
                    'sigmoid': lambda x: tf.nn.sigmoid(x),
                    'tanh': lambda x: tf.nn.tanh(x),
                    'softmax': lambda x: tf.nn.softmax(x),
                    'linear': lambda x: x
                }

            @tf.function
            def compute_layer_outputs(self, input_data):
                compute_start = time.time()
                """Efficiently compute all layer outputs in one forward pass"""
                outputs = []
                activations = []
                x = input_data
                activations.append(train_first)
                
                for layer in self.model.layers:
                    if hasattr(layer, 'activation'):
                        weights, biases = layer.kernel, layer.bias
                        linear_output = tf.matmul(x, weights) + biases
                        outputs.append(linear_output)
                        
                        activation_name = layer.activation.__name__
                        if activation_name in self.activation_functions:
                            x = self.activation_functions[activation_name](linear_output)
                        else:
                            x = linear_output
                        activations.append(x)
                    else:
                        outputs.append(x)
                        activations.append(x)
                print("Computation Time:",time.time()-compute_start)     
                return outputs, activations

            def on_train_end(self, epoch, logs=None):
                try:
                    self.update_epoch_count += 1
                    print(self.update_epoch_count)
                    if self.update_epoch_count % 10 == 0:
                        # Efficiently compute layer information
                        node_values, activated_nodes = self.compute_layer_outputs(self.train_sample)
                        
                        # Extract weights and biases efficiently
                        weights_list = []
                        biases_list = []
                        
                        # Add input layer (no weights/biases)
                        weights_list.append([[0]])
                        biases_list.append([0])
                        
                        for layer in self.model.layers:
                            if layer.get_weights():
                                w, b = layer.get_weights()
                                # Limit precision to reduce data size
                                weights_list.append(np.round(w, 4).tolist())
                                biases_list.append(np.round(b, 4).tolist())
                            else:
                                weights_list.append([[0]])
                                biases_list.append([0])

                        # Prepare detailed message
                        detailed_message = {
                            "type": "send_epoch_update",
                            "group_name": "ws_train_main",
                            "data": {
                                "epoch": self.update_epoch_count,
                                "weights": weights_list,
                                "biases": biases_list,
                                "activated_nodes": [np.round(arr.numpy(), 4).tolist() for arr in activated_nodes],
                                "loss": float(logs.get("loss", 0)),
                                "accuracy": float(logs.get("accuracy", 0)),
                            }
                        }
                        # Queue message for async sending
                        self.parent.queue_websocket_message(detailed_message)
                except Exception as e:
                    print(f"Error in WSLogger: {e}")

        def send_training_update(X_train, y_train, predictions, epoch, batch_size=500):
            """Send training updates with reduced data size"""
            try:
                # Sample data more aggressively
                sample_size = min(batch_size, len(X_train))
                indices = np.random.choice(len(X_train), sample_size, replace=False)
                X_sample = X_train[indices]
                y_sample = y_train[indices]
                pred_sample = predictions[indices]

                # Round to reduce precision and data size
                message = {
                    "type": "training_update",
                    "group_name": "ws_train_graph",
                    "data": {
                        "epoch": epoch + 1,
                        "predicted": np.round(pred_sample[:, 0] if pred_sample.shape[1] > 0 else pred_sample, 4).tolist(),
                        "x": np.round(X_sample[:, 0], 4).tolist(),
                        "y": np.round(X_sample[:, 1], 4).tolist(),
                        "labels": y_sample.tolist()
                    }
                }
                
                self.queue_websocket_message(message)
                
            except Exception as e:
                print(f"Error sending training update: {e}")

        try:
            NN_info = NeuralNetwork.objects.get(id=1)
            
            # Create optimized callback
            ws_logger = OptimizedWSLogger(self, train_first_tf)
            
            # Use tf.data for better performance with larger batch sizes
            batch_size = max(NN_info.batch_size, 64)  # Minimum batch size
            train_dataset = tf.data.Dataset.from_tensor_slices((feature_train_tf, label_train_tf))
            train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

            for epoch_num in range(0,NN_info.epoch,10):
                # Train for one epoch
                print("Epoch Num:",epoch_num)
                start_model = time.time()
                model.fit(
                    train_dataset,
                    epochs=epoch_num+10,
                    initial_epoch = epoch_num,
                    verbose=0,
                    callbacks=[ws_logger]
                )

                print("Model required:",time.time()-start_model)
                # Send graph updates even less frequently (every 10 epochs)
                predictions = fast_predict(feature_train_tf).numpy()
                send_training_update(feature_train, label_train, predictions, epoch_num)
            
        except Exception as e:
            print(f"Training error: {e}")
        finally:
            # Cleanup
            self.stop_event.set()
            if self.websocket_queue:
                self.websocket_queue.put(None)  # Signal shutdown
            if self.websocket_thread:
                self.websocket_thread.join(timeout=5)
            
        print("Training complete.")
