from django.core.management.base import BaseCommand
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
import os, warnings
import time
import gc  # For memory management
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress INFO & WARNING logs
warnings.filterwarnings("ignore")  # suppress Python warnings
import tensorflow as tf
import numpy as np
from NNvisual.training import build_model, get_dataset
from NNvisual.models import NeuralNetwork
import json

# Configure TensorFlow for low memory usage
try:
    tf.config.experimental.enable_memory_growth(True)
except:
    pass

# Pre-compute sigmoid for better performance
@tf.function
def sigmoid_tf(x):
    return tf.nn.sigmoid(x)

def sigmoid_numpy(x):
    # Clip to prevent overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

class Command(BaseCommand):
    help = "Train the neural network with periodic sync updates (Render optimized, WebSocket compatible)"

    def __init__(self):
        super().__init__()
        self.channel_layer = None
        # Remove threading components - everything runs synchronously

    def send_websocket_sync(self, message):
        """Send WebSocket messages synchronously - maintains original message format"""
        try:
            if self.channel_layer:
                async_to_sync(self.channel_layer.group_send)(
                    message["group_name"], message
                )
        except Exception as e:
            print(f"WebSocket error: {e}")

    def compress_data(self, data):
        """Compress large data payloads - same logic as original"""
        compressed = {}
        for key, value in data.items():
            if key == "weights" and isinstance(value, list):
                # More aggressive compression for low memory
                if len(value) > 2:
                    compressed[key] = [value[0], value[-1]]  # First and last only
                else:
                    compressed[key] = value
            elif key == "activated_nodes" and isinstance(value, list):
                # Sample nodes if too many - more aggressive
                if len(value) > 0 and isinstance(value[0], list) and len(value[0]) > 50:
                    compressed[key] = [arr[:25] for arr in value]  # First 25 nodes only
                else:
                    compressed[key] = value
            else:
                compressed[key] = value
        return compressed

    def create_epoch_update_message(self, epoch, model, train_sample, loss, accuracy):
        """Create the exact same epoch update message format as original"""
        try:
            # Efficiently compute layer information
            node_values = []
            activated_nodes = []
            x = train_sample
            activated_nodes.append(x)
            
            activation_functions = {
                'relu': lambda x: tf.nn.relu(x),
                'sigmoid': lambda x: tf.nn.sigmoid(x),
                'tanh': lambda x: tf.nn.tanh(x),
                'softmax': lambda x: tf.nn.softmax(x),
                'linear': lambda x: x
            }
            
            for layer in model.layers:
                if hasattr(layer, 'activation'):
                    weights, biases = layer.kernel, layer.bias
                    linear_output = tf.matmul(x, weights) + biases
                    node_values.append(linear_output)
                    
                    activation_name = layer.activation.__name__
                    if activation_name in activation_functions:
                        x = activation_functions[activation_name](linear_output)
                    else:
                        x = linear_output
                    activated_nodes.append(x)
                else:
                    node_values.append(x)
                    activated_nodes.append(x)
            
            # Extract weights and biases - same format as original
            weights_list = []
            biases_list = []
            
            # Add input layer (no weights/biases)
            weights_list.append([[0]])
            biases_list.append([0])
            
            for layer in model.layers:
                if layer.get_weights():
                    w, b = layer.get_weights()
                    # Limit precision to reduce data size
                    weights_list.append(np.round(w, 4).tolist())
                    biases_list.append(np.round(b, 4).tolist())
                else:
                    weights_list.append([[0]])
                    biases_list.append([0])

            # Prepare message in exact original format
            message_data = {
                "epoch": epoch,
                "weights": weights_list,
                "biases": biases_list,
                "activated_nodes": [np.round(arr.numpy(), 4).tolist() for arr in activated_nodes],
                "loss": float(loss),
                "accuracy": float(accuracy),
            }
            
            # Apply compression if data too large
            if len(json.dumps(message_data)) > 10000:
                message_data = self.compress_data(message_data)
            
            return {
                "type": "send_epoch_update",
                "group_name": "ws_train_main",
                "data": message_data
            }
            
        except Exception as e:
            print(f"Error creating epoch update: {e}")
            # Fallback to minimal update
            return {
                "type": "send_epoch_update",
                "group_name": "ws_train_main", 
                "data": {
                    "epoch": epoch,
                    "loss": float(loss),
                    "accuracy": float(accuracy),
                    "weights": [[[0]]],
                    "biases": [[0]],
                    "activated_nodes": [[0]]
                }
            }

    def send_training_update(self, X_train, y_train, predictions, epoch, batch_size=200):
        """Send training updates - same format as original but smaller batch"""
        try:
            # Smaller sample size for low memory
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
                    "epoch": epoch,
                    "predicted": np.round(pred_sample[:, 0] if pred_sample.shape[1] > 0 else pred_sample, 4).tolist(),
                    "x": np.round(X_sample[:, 0], 4).tolist(),
                    "y": np.round(X_sample[:, 1], 4).tolist() if X_sample.shape[1] > 1 else [],
                    "labels": y_sample.tolist()
                }
            }
            
            self.send_websocket_sync(message)
            
        except Exception as e:
            print(f"Error sending training update: {e}")

    def handle(self, *args, **kwargs):
        print("Starting Render-optimized training (max 250 epochs)...")
        
        # Get dataset once
        feature_train, label_train, feature_test, label_test, train_first, name = get_dataset()
        
        # Convert to TensorFlow tensors
        feature_train_tf = tf.constant(feature_train, dtype=tf.float32)
        label_train_tf = tf.constant(label_train, dtype=tf.float32)
        feature_test_tf = tf.constant(feature_test, dtype=tf.float32)
        label_test_tf = tf.constant(label_test, dtype=tf.float32)
        train_first_tf = tf.constant(train_first, dtype=tf.float32)
        
        model = build_model()
        
        # Setup WebSocket connection
        self.channel_layer = get_channel_layer()
        
        # Pre-compile prediction function
        @tf.function
        def fast_predict(x):
            return model(x, training=False)

        try:
            NN_info = NeuralNetwork.objects.get(id=1)
            
            # Use smaller batch size for memory efficiency
            batch_size = max(min(NN_info.batch_size, 32), 16)  # Between 16-32
            print(f"Using batch size: {batch_size}")
            
            # Create dataset with smaller batch size
            train_dataset = tf.data.Dataset.from_tensor_slices((feature_train_tf, label_train_tf))
            train_dataset = train_dataset.batch(batch_size).prefetch(1)  # Minimal prefetch
            
            # Training configuration for 250 max epochs
            max_epochs = min(NN_info.epoch, 250)  # Cap at 250
            chunk_size = 25  # Train 25 epochs at a time (10 chunks for 250 epochs)
            detailed_update_frequency = 10  # Every 10 epochs (same as original epoch%10==0 but more frequent)
            graph_update_frequency = 15    # Every 15 epochs
            
            print(f"Training {max_epochs} epochs in chunks of {chunk_size}")
            
            # Main training loop - process in chunks
            current_epoch = 0
            
            while current_epoch < max_epochs:
                # Calculate chunk boundaries
                chunk_start = current_epoch
                chunk_end = min(current_epoch + chunk_size, max_epochs)
                
                print(f"Training epochs {chunk_start} to {chunk_end}")
                chunk_start_time = time.time()
                
                # Train the chunk WITHOUT callbacks (uninterrupted training)
                history = model.fit(
                    train_dataset,
                    epochs=chunk_end,
                    initial_epoch=chunk_start,
                    verbose=0,  # No console output during training
                    # No callbacks = no interruptions during chunk training
                )
                
                chunk_time = time.time() - chunk_start_time
                current_epoch = chunk_end
                
                print(f"Chunk {chunk_start}-{chunk_end} completed in {chunk_time:.2f}s")
                
                # Get metrics from the chunk
                final_loss = history.history['loss'][-1]
                final_accuracy = history.history.get('accuracy', [0])[-1]
                
                # PERIODIC UPDATE PHASE (maintains original WebSocket format)
                update_start_time = time.time()
                
                # Send detailed epoch update (same format as original WSLogger)
                if current_epoch % detailed_update_frequency == 0 or current_epoch == max_epochs:
                    print(f"Sending epoch update for epoch {current_epoch}")
                    epoch_message = self.create_epoch_update_message(
                        current_epoch, model, train_first_tf, final_loss, final_accuracy
                    )
                    self.send_websocket_sync(epoch_message)
                
                # Send graph update for visualization (same format as original)
                if current_epoch % graph_update_frequency == 0 or current_epoch == max_epochs:
                    print(f"Sending graph update for epoch {current_epoch}")
                    
                    # Get predictions for training update
                    predictions = fast_predict(feature_train_tf).numpy()
                    self.send_training_update(feature_train, label_train, predictions, current_epoch)
                
                update_time = time.time() - update_start_time
                print(f"Updates sent in {update_time:.2f}s")
                
                # Memory cleanup after each chunk
                gc.collect()
                
                # Brief pause to let system breathe (important on shared hosting)
                time.sleep(0.1)
            
            # Send final update
            print(f"Training completed at epoch {current_epoch}")
            final_message = self.create_epoch_update_message(
                current_epoch, model, train_first_tf, final_loss, final_accuracy
            )
            self.send_websocket_sync(final_message)
            
        except Exception as e:
            print(f"Training error: {e}")
        finally:
            # Clean up memory
            gc.collect()
            print("Training completed and cleanup finished.")