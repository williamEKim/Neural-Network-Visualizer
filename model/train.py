# model/train.py
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms

# --- Activation functions ---
def sigmoid(z):
    # σ(z) = 1 / (1 + e^-z)
    return 1 / (1 + np.e ** (-z))

def sigmoid_prime(z):
    # derivative of sigmoid — you'll need this for backprop
    # σ'(z) = σ(z) * [1 - σ(z)]
    return  sigmoid(z) * (1 - sigmoid(z))

# --- Forward pass ---
def forward(x, weights, biases):
    # x is a (784,) vector
    # weights is a list of weight matrices, one per layer
    # biases is a list of bias vectors, one per layer
    # return: list of activations for EACH layer (you'll need these for backprop)
    activation_list = []
    a = x

    # add a initial activation layer to a list
    activation_list.append(a)

    for index, weight in enumerate(weights):
        a = sigmoid(weight @ a + biases[index])
        activation_list.append(a)

    return activation_list

def initialize_network(layer_sizes):
    # layer_sizes = [784, 100, 50, 16, 10]
    # return: weights (list of matrices), biases (list of vectors)
    
    weights = []
    biases = []

    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        limit = ___________________          # Xavier formula
        w = np.random.uniform(___, ___, size=(fan_out, fan_in))
        b = np.zeros(fan_out)               # biases start at 0, that's fine
        weights.append(w)
        biases.append(b)

    return weights, biases