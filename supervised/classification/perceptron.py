from deep_learning.activations import Sigmoid, ReLU, TanH, Linear
from deep_learning.loss import BCELoss, CrossEntropyLoss, MSELoss, L1Loss
import numpy as np


class Perceptron(object):
    def __init__(self, loss_function=BCELoss, activation_function=Sigmoid, learning_rate=.01, max_iterations=5000):
        self.learning_rate = learning_rate
        self.loss_function = loss_function()
        self.activation_function = activation_function()
        self.max_iterations = max_iterations
        self.is_fitted = False
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # Count the number of samples
        n_samples, n_features = X.shape
        _, n_outputs = y.shape

        # Initialise the weights
        self.weights = np.random.uniform(-.5, .5, (n_features, n_outputs))
        self.bias = np.ones((1, n_outputs))

        for _ in range(self.max_iterations):
            # Prediction
            raw_prediction = self.__raw_prediction(X)
            y_pred = self.activation_function(raw_prediction)

            # Chain rule
            loss_gradient = self.loss_function.gradient(y, y_pred)
            activation_gradient = self.activation_function.gradient(raw_prediction)
            error_term = loss_gradient * activation_gradient

            self.weights -= self.learning_rate * X.T.dot(error_term)
            self.bias -= self.learning_rate * np.mean(error_term, axis=0, keepdims=True)

    def __raw_prediction(self, X):
        return X.dot(self.weights) + self.bias

    def predict(self, X):
        return self.activation_function(X.dot(self.weights) + self.bias)
