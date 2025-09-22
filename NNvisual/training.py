from tensorflow.keras.models import Sequential # type: ignore 
from tensorflow.keras.layers import Dense # type: ignore 
from tensorflow.keras.optimizers import Adam # type: ignore 
import tensorflow as tf
from .models import NeuralNetwork


def get_dataset():
    NN_info = NeuralNetwork.objects.get(id=1)

    if NN_info.dataset == 1:
        name = ("Moons")
        from sklearn.datasets import make_moons
        features, labels = make_moons(n_samples = 500, noise = 0.3, random_state=42)
    elif NN_info.dataset == 2:
        name = ("Circles")
        from sklearn.datasets import make_circles
        features, labels = make_circles(n_samples = 500, noise = 0.1, random_state=42)
    elif NN_info.dataset == 4:
        name = ("Random")
        from sklearn.datasets import make_classification
        features, labels = make_classification(n_samples=500, n_features=2, 
                           n_informative=2, n_redundant=0, 
                           n_clusters_per_class=2, random_state=42)
    elif NN_info.dataset == 3:
        name = ("Blobs")
        from sklearn.datasets import make_blobs
        features, labels = make_blobs(n_samples=500, centers=2, n_features=2, random_state=42)

    n = 80
    feature_train = features[:n]
    label_train = labels[:n]
    feature_test = features[n:]
    label_test = labels[n:]
    train_first = feature_train[0].reshape(1, 2)

    return feature_train, label_train, feature_test, label_test, train_first, name

def build_model():
    NN_info = NeuralNetwork.objects.get(id=1)
    model = Sequential([
        Dense(8 , input_shape=(2,), activation=NN_info.activation_function),
        Dense(8 , activation=NN_info.activation_function),
        Dense(1 , activation='sigmoid')
    ])

    
    model.compile(optimizer=Adam(learning_rate=NN_info.learning_rate),
                loss='binary_crossentropy',
                metrics=['accuracy'])
    model(tf.zeros((1,2))) 
    return model

