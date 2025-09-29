from .models import NeuralNetwork

NN_config = {
    "User" : {
        "epoch" : 100,
        "batch_size" : 30,
        "learning_rate" : 0.01,
        "activation_function" : "relu",
        "dataset" : 2
    }
}

# ACTIVATION_CHOICES = [
#         ('relu', 'ReLU'),
#         ('sigmoid', 'Sigmoid'),
#         ('tanh', 'Tanh'),
#         ('linear', 'Linear'),
#     ]

# Dataset_Mapping = [

# ]