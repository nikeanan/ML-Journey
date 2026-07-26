"""
Neural Network implementation from scratch using vectorized NumPy.
Includes Dense layers, ReLU, Sigmoid, and Softmax activation functions.
"""

from typing import List, Tuple
import numpy as np


def sigmoid(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    return a, z


def sigmoid_backward(da: np.ndarray, z: np.ndarray) -> np.ndarray:
    s = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    return da * s * (1.0 - s)


def relu(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.maximum(0, z)
    return a, z


def relu_backward(da: np.ndarray, z: np.ndarray) -> np.ndarray:
    dz = np.array(da, copy=True)
    dz[z <= 0] = 0
    return dz


class DenseLayer:
    """
    Fully-connected Dense neural network layer.
    """

    def __init__(self, input_dim: int, output_dim: int, activation: str = "relu"):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation.lower()

        # He initialization for ReLU, Xavier for Sigmoid
        scale = np.sqrt(2.0 / input_dim) if self.activation == "relu" else np.sqrt(1.0 / input_dim)
        self.W = np.random.randn(input_dim, output_dim) * scale
        self.b = np.zeros((1, output_dim))

        self.dW = None
        self.db = None
        self.cache = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        Z = np.dot(X, self.W) + self.b
        if self.activation == "relu":
            A, activation_cache = relu(Z)
        elif self.activation == "sigmoid":
            A, activation_cache = sigmoid(Z)
        else:
            A, activation_cache = Z, Z

        self.cache = (X, Z)
        return A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        X, Z = self.cache
        m = X.shape[0]

        if self.activation == "relu":
            dZ = relu_backward(dA, Z)
        elif self.activation == "sigmoid":
            dZ = sigmoid_backward(dA, Z)
        else:
            dZ = dA

        self.dW = (1.0 / m) * np.dot(X.T, dZ)
        self.db = (1.0 / m) * np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.W.T)

        return dX


class NeuralNetwork:
    """
    Multi-Layer Perceptron (MLP) Sequential Neural Network container.
    """

    def __init__(self):
        self.layers: List[DenseLayer] = []

    def add(self, layer: DenseLayer):
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 1000,
        learning_rate: float = 0.01,
        verbose: bool = False,
    ) -> List[float]:
        m = X.shape[0]
        y_reshaped = y.reshape(-1, 1) if y.ndim == 1 else y
        cost_history = []

        for epoch in range(epochs):
            # Forward pass
            A_out = self.forward(X)

            # Binary Cross Entropy Cost
            cost = - (1.0 / m) * np.sum(
                y_reshaped * np.log(np.clip(A_out, 1e-15, 1 - 1e-15))
                + (1 - y_reshaped) * np.log(np.clip(1 - A_out, 1e-15, 1 - 1e-15))
            )
            cost_history.append(float(cost))

            # Initial gradient
            dA = -(y_reshaped / np.clip(A_out, 1e-15, 1.0)) + (
                (1 - y_reshaped) / np.clip(1 - A_out, 1e-15, 1.0)
            )

            # Backward pass
            for layer in reversed(self.layers):
                dA = layer.backward(dA)

            # Gradient update
            for layer in self.layers:
                layer.W -= learning_rate * layer.dW
                layer.b -= learning_rate * layer.db

            if verbose and epoch % (epochs // 10) == 0:
                print(f"Epoch {epoch:5d} - Binary Cross-Entropy Cost: {cost:.5f}")

        return cost_history
