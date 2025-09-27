import torch
import asyncio
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

    def forward(self, input, epoch, log_interval=10):
        self.data = []
        first_nodes = func.relu(self.fc1(input))
        second_nodes = func.relu(self.fc2(first_nodes))
        output = self.output(second_nodes)

        # Only collect data when needed
        if epoch % log_interval == 0:
            with torch.no_grad():  # No gradients needed for logging
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
        
        # Threading control - no queue, just direct sending
        self.active_send_threads = []
        self.max_concurrent_sends = 3  # Limit concurrent sends
        self.send_lock = threading.Lock()
        
        # Timing statistics
        self.timing_stats = {
            'forward_time': [],
            'backward_time': [],
            'message_prep_time': [],
            'total_epoch_time': [],
            'send_thread_count': 0
        }

    def send_web_data_threaded(self, message):
        """Clean threading without queue - waits if too many concurrent sends"""
        def worker(msg, thread_id):
            send_start = time.time()
            try:
                channel_layer = get_channel_layer()
                if channel_layer:
                    async_to_sync(channel_layer.group_send)(
                        msg["group_name"], msg
                    )
                send_time = time.time() - send_start
                print(f"  └─ Send thread {thread_id} completed in {send_time:.3f}s")
            except Exception as e:
                print(f"  └─ Send thread {thread_id} error: {e}")
            finally:
                # Remove this thread from active list
                with self.send_lock:
                    self.active_send_threads = [t for t in self.active_send_threads if t != thread_id]

        # Wait if too many concurrent sends (prevents overwhelming frontend)
        while len(self.active_send_threads) >= self.max_concurrent_sends:
            time.sleep(0.001)  # Brief wait

        with self.send_lock:
            thread_id = self.timing_stats['send_thread_count']
            self.timing_stats['send_thread_count'] += 1
            self.active_send_threads.append(thread_id)

        print(f"  ├─ Starting send thread {thread_id} (active: {len(self.active_send_threads)})")
        threading.Thread(target=worker, args=(message, thread_id), daemon=True).start()

    async def train(self):
        """Training loop with detailed timing"""
        log_interval = 5  # Send every 5 epochs for better sync
        
        print(f"Starting training for {self.epoch} epochs with log_interval={log_interval}")
        print(f"Max concurrent sends: {self.max_concurrent_sends}")
        print("-" * 60)
        
        for i in range(self.epoch):
            epoch_start = time.time()
            
            # Forward pass timing
            forward_start = time.time()
            predictions, data = self.model.forward(values, i, log_interval)
            loss = self.criterion(predictions, labels)
            forward_time = time.time() - forward_start
            
            self.losses.append(loss.item())

            # Message preparation timing
            if i % log_interval == 0 and data:
                prep_start = time.time()
                
                with torch.no_grad():
                    pred = torch.sigmoid(predictions) > 0.5
                    accuracy = (pred == labels).float().mean().item()

                    # Extract weights and biases
                    weights = [[[0]]]
                    biases = [[0]]
                    
                    for name, param in self.model.named_parameters():
                        param_cpu = param.detach().cpu()
                        if "weight" in name:
                            weights.append([[round(float(v), 2) for v in row] for row in param_cpu])
                        elif "bias" in name:
                            biases.append([round(float(v), 2) for v in param_cpu])

                    message = self.create_message(i, weights, biases, data, loss, accuracy)
                    
                prep_time = time.time() - prep_start
                self.timing_stats['message_prep_time'].append(prep_time)
                
                # Start sending in background (non-blocking but controlled)
                print(f"Epoch {i}: Preparing to send message...")
                self.send_web_data_threaded(message)
            
            # Backward pass timing
            backward_start = time.time()
            self.optimized.zero_grad()
            loss.backward()
            self.optimized.step()
            backward_time = time.time() - backward_start
            
            # Total epoch timing
            total_epoch_time = time.time() - epoch_start
            
            # Store timing stats
            self.timing_stats['forward_time'].append(forward_time)
            self.timing_stats['backward_time'].append(backward_time)
            self.timing_stats['total_epoch_time'].append(total_epoch_time)
            
            # Print detailed timing info
            if i % log_interval == 0:
                active_sends = len(self.active_send_threads)
                print(f"Epoch {i}:")
                print(f"  ├─ Loss: {loss.item():.4f}, Accuracy: {accuracy:.3f}" if 'accuracy' in locals() else f"  ├─ Loss: {loss.item():.4f}")
                print(f"  ├─ Forward: {forward_time*1000:.1f}ms, Backward: {backward_time*1000:.1f}ms")
                print(f"  ├─ Message prep: {prep_time*1000:.1f}ms" if i % log_interval == 0 and data else "  ├─ No message sent")
                print(f"  ├─ Total epoch: {total_epoch_time*1000:.1f}ms")
                print(f"  └─ Active send threads: {active_sends}")
                print()
            else:
                # Simplified output for non-logging epochs
                print(f"Epoch {i}: {total_epoch_time*1000:.1f}ms (Forward: {forward_time*1000:.1f}ms, Backward: {backward_time*1000:.1f}ms)")

        # Wait for all send threads to complete
        print("Training completed. Waiting for remaining send threads...")
        while self.active_send_threads:
            time.sleep(0.1)
            print(f"Waiting for {len(self.active_send_threads)} send threads to complete...")
        
        self.print_timing_summary()

    def print_timing_summary(self):
        """Print comprehensive timing statistics"""
        print("\n" + "="*60)
        print("TIMING SUMMARY")
        print("="*60)
        
        avg_forward = sum(self.timing_stats['forward_time']) / len(self.timing_stats['forward_time'])
        avg_backward = sum(self.timing_stats['backward_time']) / len(self.timing_stats['backward_time'])
        avg_total = sum(self.timing_stats['total_epoch_time']) / len(self.timing_stats['total_epoch_time'])
        
        print(f"Average Forward Pass:  {avg_forward*1000:.2f}ms")
        print(f"Average Backward Pass: {avg_backward*1000:.2f}ms")
        print(f"Average Total Epoch:   {avg_total*1000:.2f}ms")
        
        if self.timing_stats['message_prep_time']:
            avg_prep = sum(self.timing_stats['message_prep_time']) / len(self.timing_stats['message_prep_time'])
            print(f"Average Message Prep:  {avg_prep*1000:.2f}ms")
        
        print(f"Total Send Threads Created: {self.timing_stats['send_thread_count']}")
        print(f"Training Efficiency: {(avg_forward + avg_backward)/avg_total*100:.1f}% (training vs total time)")

    def create_message(self, epoch, weights, biases, nodes, loss, accuracy=1):
        """Create WebSocket message"""
        message_data = {
            "epoch": epoch,
            "weights": weights,
            "biases": biases,
            "activated_nodes": nodes,
            "loss": float(loss.item() if hasattr(loss, 'item') else loss),
            "accuracy": float(accuracy),
        }
        return {
            "type": "send_epoch_update",
            "group_name": "ws_train_main",
            "data": message_data
        }

    def cleanup(self):
        """Wait for all background threads to complete"""
        print("Cleaning up...")
        while self.active_send_threads:
            print(f"Waiting for {len(self.active_send_threads)} active send threads...")
            time.sleep(0.5)
        print("Cleanup completed!")

# Usage example:
# trainer = TrainModel(epoch=100)
# await trainer.train()
# trainer.cleanup()