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