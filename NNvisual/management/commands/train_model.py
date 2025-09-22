from NNvisual.training import build_model, get_dataset
from django.core.management.base import BaseCommand
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
import tensorflow as tf
import numpy as np
import time
from NNvisual.models import NeuralNetwork
import threading
from queue import Queue

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
        self.websocket_queue = Queue()
        self.websocket_thread = None
        self.stop_event = threading.Event()

    def setup_websocket_handler(self):
        """Setup async websocket message handler in separate thread"""
        def websocket_worker():
            while not self.stop_event.is_set():
                try:
                    message = self.websocket_queue.get(timeout=0.1)
                    if message is None:  # Shutdown signal
                        break
                    async_to_sync(self.channel_layer.group_send)(
                        "neural_network_updates", message
                    )
                    self.websocket_queue.task_done()
                except:
                    continue
        
        self.websocket_thread = threading.Thread(target=websocket_worker)
        self.websocket_thread.start()

    def queue_websocket_message(self, message):
        """Queue websocket message for async sending"""
        try:
            self.websocket_queue.put_nowait(message)
        except:
            # Queue full, skip this update
            pass

    def handle(self, *args, **kwargs):

        # Get dataset once
        NeuralNetwork.objects.get_or_create(
            id=1,  # or any unique field to check existence
            defaults={
                'epoch': 100,
                'batch_size': 30,
                'learning_rate': 0.01,
                'activation_function':'tanh',
                'dataset':1
            }
        )
        feature_train, label_train, feature_test, label_test, train_first, name = get_dataset()
        print("Model Name:",name)
        
        # Convert to TensorFlow tensors for better performance
        feature_train_tf = tf.constant(feature_train, dtype=tf.float32)
        label_train_tf = tf.constant(label_train, dtype=tf.float32)
        feature_test_tf = tf.constant(feature_test, dtype=tf.float32)
        label_test_tf = tf.constant(label_test, dtype=tf.float32)
        train_first_tf = tf.constant(train_first, dtype=tf.float32)
        start = time.time()
        model = build_model()
        print("Model Built in:",time.time()-start)

        # Setup WebSocket handling
        self.channel_layer = get_channel_layer()
        self.setup_websocket_handler()

        # Pre-compile prediction function
        @tf.function
        def fast_predict(x):
            return model(x, training=False)

        class OptimizedWSLogger(tf.keras.callbacks.Callback):
            def __init__(self, parent_command, train_sample,  update_epoch_count=0):
                super().__init__()
                self.parent = parent_command
                self.train_sample = train_sample
                self.update_epoch_count = update_epoch_count
                self.activation_functions = {
                    'relu': lambda x: tf.nn.relu(x),
                    'sigmoid': lambda x: tf.nn.sigmoid(x),
                    'tanh': lambda x: tf.nn.tanh(x),
                    'softmax': lambda x: tf.nn.softmax(x),
                    'linear': lambda x: x
                }

            @tf.function
            def compute_layer_outputs(self, input_data):
                """Efficiently compute all layer outputs in one forward pass"""
                outputs = []
                activations = []
                x = input_data
                activations.append(train_first)
                for layer in self.model.layers:
                    # Get layer output before activation
                    if hasattr(layer, 'activation'):
                        # Manually compute linear transformation
                        weights, biases = layer.kernel, layer.bias
                        linear_output = tf.matmul(x, weights) + biases
                        outputs.append(linear_output)
                        
                        # Apply activation
                        activation_name = layer.activation.__name__
                        if activation_name in self.activation_functions:
                            x = self.activation_functions[activation_name](linear_output)
                        else:
                            x = linear_output
                        activations.append(x)
                    else:
                        outputs.append(x)
                        activations.append(x)
                        
                return outputs, activations

            def on_epoch_end(self, epoch, logs=None):
                    
                try:
                    self.update_epoch_count+=1
                    if self.update_epoch_count%2==0:
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
                                weights_list.append(w.tolist())
                                biases_list.append(b.tolist())
                            else:
                                weights_list.append([[0]])
                                biases_list.append([0])

                        # Prepare message
                        message = {
                            "type": "send_epoch_update",
                            "data": {
                                "epoch": self.update_epoch_count,
                                "weights": weights_list,
                                "biases": biases_list,
                                "activated_nodes": [arr.numpy().tolist() for arr in activated_nodes],
                                "node_values": [arr.numpy().tolist() for arr in node_values],
                                "loss": float(logs.get("loss", 0)),
                                "accuracy": float(logs.get("accuracy", 0)),
                                "val_loss": float(logs.get("val_loss", 0)),
                                "val_accuracy": float(logs.get("val_accuracy", 0))
                            }
                        }
                        
                        # Queue message for async sending
                        self.parent.queue_websocket_message(message)
                        
                except Exception as e:
                    print(f"Error in WSLogger: {e}")

        def send_training_update(X_train, y_train, predictions, epoch, batch_size=1000):
            """Send training updates in batches to reduce memory usage"""
            try:
                # Sample data if too large
                if len(X_train) > batch_size:
                    indices = np.random.choice(len(X_train), batch_size, replace=False)
                    X_sample = X_train[indices]
                    y_sample = y_train[indices]
                    pred_sample = predictions[indices]
                else:
                    X_sample = X_train
                    y_sample = y_train
                    pred_sample = predictions

                message = {
                    "type": "training_update",
                    "data": {
                        "epoch": epoch + 1,
                        "predicted": pred_sample[:, 0].tolist() if pred_sample.shape[1] > 0 else pred_sample.tolist(),
                        "x": X_sample[:, 0].tolist(),
                        "y": X_sample[:, 1].tolist(),
                        "labels": y_sample.tolist()
                    }
                }
                
                self.queue_websocket_message(message)
                
            except Exception as e:
                print(f"Error sending training update: {e}")

        try:
            NN_info = NeuralNetwork.objects.get(id=1)
            
            # Create optimized callback
            ws_logger = OptimizedWSLogger(self, train_first_tf,)
            
            # Use tf.data for better performance
            train_dataset = tf.data.Dataset.from_tensor_slices((feature_train_tf, label_train_tf))
            train_dataset = train_dataset.batch(NN_info.batch_size).prefetch(tf.data.AUTOTUNE)
            
            val_dataset = tf.data.Dataset.from_tensor_slices((feature_test_tf, label_test_tf))
            val_dataset = val_dataset.batch(NN_info.batch_size).prefetch(tf.data.AUTOTUNE)

            for epoch in range(NN_info.epoch):
                # Train for one epoch
                print(epoch)
                model.fit(
                    train_dataset,
                    epochs=1,
                    # validation_data=val_dataset,
                    verbose=0,
                    callbacks=[ws_logger]
                )
                if epoch%1==0:
                    predictions = fast_predict(feature_train_tf).numpy()
                    send_training_update(feature_test, label_test, predictions, epoch)
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
