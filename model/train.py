# model/train.py
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms

# -------------------- Activation functions --------------------
def sigmoid(z):
    # σ(z) = 1 / (1 + e^-z)
    return 1 / (1 + np.e ** (-z))

def sigmoid_prime(z):
    # derivative of sigmoid — you'll need this for backprop
    # σ'(z) = σ(z) * [1 - σ(z)]
    return  sigmoid(z) * (1 - sigmoid(z))

# -------------------- Forward pass --------------------
def forward(x, weights, biases):
    # x is a (784,) vector
    # weights is a list of weight matrices, one per layer
    # biases is a list of bias vectors, one per layer
    # return: list of activations for EACH layer AND z values(you'll need these for backprop)
    activation_list = []
    z_list = []
    a = x

    # add a initial activation layer to a list
    activation_list.append(a)

    for index, weight in enumerate(weights):
        z = weight @ a + biases[index]
        a = sigmoid(z)
        activation_list.append(a)
        z_list.append(z)

    return activation_list, z_list


def initialize_network(layer_sizes):
    # layer_sizes = [784, 100, 50, 16, 10]
    # return: weights (list of matrices), biases (list of vectors)
    
    weights = []
    biases = []

    for fan_in, fan_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        limit = np.sqrt(6 / (fan_in + fan_out))          # Xavier formula
        w = np.random.uniform(-limit, limit, size=(fan_out, fan_in))
        b = np.zeros(fan_out)                            # biases start at 0, that's fine
        weights.append(w)
        biases.append(b)

    return weights, biases


def one_hot(label, num_classes=10):
    # label is an int 0-9
    # return a (10,) vector of zeros with a 1 at index label
    v = np.zeros(num_classes)
    v[label] = 1.0
    return v

def mse_loss(y_pred, y_true):
    # y_pred is the output activation (10,) vector
    # y_true is the one-hot encoded label (10,) vector
    # L = (1/n) * Σ (y_pred - y_true)^2
    # return a scalar loss value
    return (1/len(y_pred)) * np.sum((y_pred - y_true) ** 2)


def backward(x, y_true, activations, z_list, weights):
    # x           — original input (784,)
    # y_true      — one-hot label (10,)
    # activations — list from forward(), one array per layer
    # weights     — list of weight matrices
    # biases      — list of bias vectors
    # return: grad_weights, grad_biases (same structure as weights, biases)

    grad_weights = []
    grad_biases = []

    # Step 1 — output layer delta
    delta = (activations[-1] - y_true) * sigmoid_prime(z_list[-1])     # delta = ∂L/∂a * sigmoid_prime(z)

    # Step 2 — loop backward through layers
    for i in reversed(range(len(weights))):
        a_prev = activations[i]  # activation feeding INTO this layer

        # Step 3 — compute gradients for this layer
        grad_w = np.outer(delta, a_prev)  # outer product of delta and a_prev
        grad_b = delta

        grad_weights.append(grad_w)
        grad_biases.append(grad_b)

        # pass delta backward (skip on last iteration — no layer before input)
        if i > 0:
            delta = weights[i].T @ delta * sigmoid_prime(z_list[i-1])

    # reverse since we appended back-to-front
    grad_weights.reverse()
    grad_biases.reverse()

    return grad_weights, grad_biases


# -------------------- Training --------------------
def train(weights, biases, x_train, y_train, epochs=10, lr=0.01, batch_size=32):
    for epoch in range(epochs):
        # Step 1 — shuffle the data at the start of each epoch
        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]
        y_train = y_train[indices]

        # Step 2 — split into mini-batches
        for batch_start in range(0, len(x_train), batch_size):
            x_batch = x_train[batch_start : batch_start + batch_size]
            y_batch = y_train[batch_start : batch_start + batch_size]

            # Step 3 — accumulate gradients across the batch
            grad_w_total = [np.zeros_like(w) for w in weights]  # same structure as weights, but all zeros
            grad_b_total = [np.zeros_like(b) for b in biases]  # same structure as biases, but all zeros

            for x, y in zip(x_batch, y_batch):
                y_enc = one_hot(y)
                activations, z_list = forward(x, weights, biases)  # forward pass
                gw, gb = backward(x, y_enc, activations, z_list, weights)              # backward pass

                # accumulate
                for i in range(len(weights)):
                    grad_w_total[i] += gw[i]
                    grad_b_total[i] += gb[i]

            # Step 4 — average and update
            for i in range(len(weights)):
                weights[i] -= lr * (grad_w_total[i] / len(x_batch))
                biases[i]  -= lr * (grad_b_total[i] / len(x_batch))

        print(f"Epoch {epoch+1}/{epochs} done")

    return weights, biases