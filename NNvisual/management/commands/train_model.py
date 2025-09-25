from django.core.management.base import BaseCommand
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
import os, warnings
import time
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")
import tensorflow as tf
import numpy as np
from NNvisual.training import build_model, get_dataset
from NNvisual.models import NeuralNetwork
import threading
from queue import Queue, Empty
import json

class Command(BaseCommand):
    help = "Train the neural network and broadcast updates"

    def __init__(self):
        super().__init__()
        self.channel_layer = None
        self.websocket_queue = Queue(maxsize=20)  # Smaller queue
        self.websocket_thread = None
        self.stop_event = threading.Event()
        self.last_update_time = 0
        self.min_update_interval = 0.2  # Less frequent updates (200ms)

    def setup_websocket_handler(self):
        """Optimized async websocket message handler"""
        def websocket_worker():
            batch_messages = []
            batch_timeout = 0.1
            last_batch_send = time.time()
            
            while not self.stop_event.is_set():
                try:
                    message = self.websocket_queue.get(timeout=batch_timeout)
                    if message is None:
                        break
                    
                    batch_messages.append(message)
                    
                    # Send batch when queue is empty or timeout reached
                    current_time = time.time()
                    if (self.websocket_queue.empty() or 
                        current_time - last_batch_send > batch_timeout):
                        
                        for msg in batch_messages:
                            try:
                                async_to_sync(self.channel_layer.group_send)(
                                    msg.get("group_name"), msg
                                )
                            except Exception as e:
                                print(f"WebSocket send error: {e}")
                        
                        batch_messages.clear()
                        last_batch_send = current_time
                        
                    self.websocket_queue.task_done()
                    
                except Empty:
                    # Send any pending batched messages
                    if batch_messages:
                        for msg in batch_messages:
                            try:
                                async_to_sync(self.channel_layer.group_send)(
                                    msg.get("group_name"), msg
                                )
                            except:
                                pass
                        batch_messages.clear()
                    continue
                except Exception as e:
                    print(f"WebSocket error: {e}")
                    continue
        
        self.websocket_thread = threading.Thread(target=websocket_worker, daemon=True)
        self.websocket_thread.start()

    def queue_websocket_message(self, message):
        """Optimized message queuing with aggressive rate limiting"""
        current_time = time.time()
        
        if current_time - self.last_update_time < self.min_update_interval:
            return
            
        try:
            self.websocket_queue.put_nowait(message)
            self.last_update_time = current_time
        except:
            pass

    def handle(self, *args, **kwargs):
        print("Train Command called")
        
        # Get configuration first
        NN_info = NeuralNetwork.objects.get(id=1)
        
        # Get dataset
        feature_train, label_train, feature_test, label_test, train_first, name = get_dataset()
        
        # Use smaller sample for training if dataset is large
        max_samples = 5000  # Limit training samples
        if len(feature_train) > max_samples:
            indices = np.random.choice(len(feature_train), max_samples, replace=False)
            feature_train = feature_train[indices]
            label_train = label_train[indices]
        
        # Convert to TensorFlow tensors
        feature_train_tf = tf.constant(feature_train, dtype=tf.float32)
        label_train_tf = tf.constant(label_train, dtype=tf.float32)
        train_first_tf = tf.constant(train_first, dtype=tf.float32)
        
        print("Build started", start := time.time())
        model = build_model()
        print("Build ended", time.time()-start)
        
        # Setup WebSocket handling
        self.channel_layer = get_channel_layer()
        self.setup_websocket_handler()

        # Pre-compile prediction function
        @tf.function(experimental_relax_shapes=True)
        def fast_predict(x):
            return model(x, training=False)

        class SuperOptimizedWSLogger(tf.keras.callbacks.Callback):
            def __init__(self, parent_command, train_sample):
                super().__init__()
                self.parent = parent_command
                self.train_sample = train_sample
                self.epoch_count = 0
                self.last_detailed_update = 0
                
                # Pre-compute activation functions
                self.activation_map = {
                    'relu': tf.nn.relu,
                    'sigmoid': tf.nn.sigmoid,
                    'tanh': tf.nn.tanh,
                    'softmax': tf.nn.softmax,
                    'linear': tf.identity
                }

            def on_epoch_end(self, epoch, logs=None):
                try:
                    self.epoch_count += 1
                    
                    # Only send detailed updates every 20 epochs (less frequent)
                    if self.epoch_count % 20 == 0:
                        self.send_detailed_update(logs)
                    
                    # Send basic metrics every 5 epochs
                    elif self.epoch_count % 5 == 0:
                        self.send_basic_update(logs)
                        
                except Exception as e:
                    print(f"Error in WSLogger: {e}")

            def send_basic_update(self, logs):
                """Send lightweight updates"""
                message = {
                    "type": "send_epoch_update",
                    "group_name": "ws_train_main",
                    "data": {
                        "epoch": self.epoch_count,
                        "loss": float(logs.get("loss", 0)),
                        "accuracy": float(logs.get("accuracy", 0)),
                        "basic": True
                    }
                }
                self.parent.queue_websocket_message(message)

            @tf.function(experimental_relax_shapes=True)
            def compute_activations_fast(self, input_data):
                """Optimized activation computation"""
                activations = [input_data]
                x = input_data
                
                for layer in self.model.layers:
                    if hasattr(layer, 'kernel'):
                        x = tf.matmul(x, layer.kernel) + layer.bias
                        if hasattr(layer, 'activation') and layer.activation is not None:
                            if hasattr(layer.activation, '__name__'):
                                activation_name = layer.activation.__name__
                                if activation_name in self.activation_map:
                                    x = self.activation_map[activation_name](x)
                        activations.append(x)
                
                return activations

            def send_detailed_update(self, logs):
                """Send detailed updates less frequently"""
                start_time = time.time()
                
                # Compute activations efficiently
                activations = self.compute_activations_fast(self.train_sample)
                
                # Extract only essential weight information
                weights_list = []
                biases_list = []
                
                # Simplified weight extraction
                for layer in self.model.layers:
                    if layer.get_weights():
                        w, b = layer.get_weights()
                        # Reduce precision and sample weights if too large
                        if w.size > 1000:  # Sample large weight matrices
                            w_sample = w.flatten()
                            indices = np.linspace(0, len(w_sample)-1, 100, dtype=int)
                            w_reduced = w_sample[indices].reshape(-1, 1)
                        else:
                            w_reduced = w
                        
                        weights_list.append(np.round(w_reduced, 3).tolist())
                        biases_list.append(np.round(b[:10], 3).tolist())  # First 10 biases only
                    else:
                        weights_list.append([[0]])
                        biases_list.append([0])

                message = {
                    "type": "send_epoch_update", 
                    "group_name": "ws_train_main",
                    "data": {
                        "epoch": self.epoch_count,
                        "weights": weights_list,
                        "biases": biases_list,
                        "activated_nodes": [np.round(arr.numpy()[:50], 3).tolist() for arr in activations],  # First 50 nodes
                        "loss": float(logs.get("loss", 0)),
                        "accuracy": float(logs.get("accuracy", 0)),
                    }
                }
                
                print(f"Detailed update time: {time.time() - start_time:.3f}s")
                self.parent.queue_websocket_message(message)

        def send_training_update_optimized(X_train, y_train, predictions, epoch):
            """Ultra-lightweight training updates"""
            try:
                # Use much smaller sample
                sample_size = min(200, len(X_train))  # Reduced from 500
                indices = np.random.choice(len(X_train), sample_size, replace=False)
                
                message = {
                    "type": "training_update",
                    "group_name": "ws_train_graph", 
                    "data": {
                        "epoch": epoch + 1,
                        "predicted": np.round(predictions[indices, 0], 3).tolist(),
                        "x": np.round(X_train[indices, 0], 3).tolist(),
                        "y": np.round(X_train[indices, 1], 3).tolist(),
                        "labels": y_train[indices].tolist()
                    }
                }
                
                self.queue_websocket_message(message)
                
            except Exception as e:
                print(f"Error sending training update: {e}")

        try:
            # Create optimized callback
            ws_logger = SuperOptimizedWSLogger(self, train_first_tf)
            
            # Optimize batch size - larger batches = fewer iterations
            optimal_batch_size = max(NN_info.batch_size, 128)  # Increased minimum
            
            # Create optimized dataset pipeline
            train_dataset = tf.data.Dataset.from_tensor_slices((feature_train_tf, label_train_tf))
            train_dataset = (train_dataset
                           .cache()  # Cache in memory
                           .shuffle(buffer_size=min(1000, len(feature_train)))
                           .batch(optimal_batch_size, drop_remainder=True)
                           .prefetch(tf.data.AUTOTUNE))
            
            # **KEY OPTIMIZATION: Use single training call with initial_epoch**
            total_epochs = NN_info.epoch
            update_frequency = max(1, total_epochs // 20)  # Update 20 times max
            
            start_total_training = time.time()
            
            for epoch_batch in range(0, total_epochs, 10):
                batch_end = min(epoch_batch + 10, total_epochs)
                
                start_model = time.time()
                
                # Continue training from where we left off
                model.fit(
                    train_dataset,
                    epochs=batch_end,
                    initial_epoch=epoch_batch,  # KEY: Don't restart training
                    verbose=0,
                    callbacks=[ws_logger] if epoch_batch % 20 == 0 else []  # Callbacks only occasionally
                )
                
                print(f"Epochs {epoch_batch}-{batch_end} required: {time.time()-start_model:.3f}s")
                
                # Send graph updates much less frequently
                if epoch_batch % 40 == 0:  # Every 40 epochs instead of 10
                    predictions = fast_predict(feature_train_tf).numpy()
                    send_training_update_optimized(feature_train, label_train, predictions, epoch_batch)
            
            print(f"Total training time: {time.time() - start_total_training:.3f}s")
            
        except Exception as e:
            print(f"Training error: {e}")
        finally:
            # Cleanup
            self.stop_event.set()
            if self.websocket_queue:
                self.websocket_queue.put(None)
            if self.websocket_thread:
                self.websocket_thread.join(timeout=2)
            
        print("Training complete.")