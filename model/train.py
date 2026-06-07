# model/train.py
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms

# --- Activation functions ---
def sigmoid(z):
    pass  # TODO

def sigmoid_prime(z):
    # derivative of sigmoid — you'll need this for backprop
    pass  # TODO

# --- Forward pass ---
def forward(x, weights, biases):
    # x is a (784,) vector
    # weights is a list of weight matrices, one per layer
    # biases is a list of bias vectors, one per layer
    # return: list of activations for EACH layer (you'll need these for backprop)
    pass  # TODO