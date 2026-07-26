"""
Unit tests for neural network module built from scratch.
"""

import numpy as np
import pytest
from ml_journey.neural_net import DenseLayer, NeuralNetwork, relu, sigmoid


def test_activations():
    z = np.array([-2.0, 0.0, 2.0])
    a_relu, _ = relu(z)
    assert np.allclose(a_relu, np.array([0.0, 0.0, 2.0]))

    a_sig, _ = sigmoid(z)
    assert np.allclose(a_sig, 1.0 / (1.0 + np.exp(-z)))


def test_neural_network_forward():
    np.random.seed(42)
    X = np.random.randn(10, 4)

    nn = NeuralNetwork()
    nn.add(DenseLayer(4, 8, activation="relu"))
    nn.add(DenseLayer(8, 1, activation="sigmoid"))

    output = nn.forward(X)
    assert output.shape == (10, 1)
    assert np.all((output >= 0.0) & (output <= 1.0))


def test_neural_network_training():
    np.random.seed(42)
    # Simple XOR problem
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)

    nn = NeuralNetwork()
    nn.add(DenseLayer(2, 4, activation="relu"))
    nn.add(DenseLayer(4, 1, activation="sigmoid"))

    history = nn.fit(X, y, epochs=500, learning_rate=0.1)

    assert len(history) == 500
    assert history[-1] < history[0]  # Cost decreased
