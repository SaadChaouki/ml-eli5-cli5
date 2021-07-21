import numpy as np
import math


class NaiveBayesClassifier(object):
    def __init__(self):
        self.X = None
        self.y = None
        self.classes = None
        self.params = {}
        self.priors = {}

    def fit(self, X, y):
        # Store the data needed
        self.X, self.y, self.classes = X, y, np.unique(y)

        # Calculate the priors and the mean/variance for each class
        for c in self.classes:
            # Compute the prior for the class
            self.priors[c] = np.mean(self.y == c)
            self.params[c] = {}

            # Select data of the class
            X_c = self.X[self.y == c]

            for i in range(self.X.shape[1]):
                self.params[c][i] = {'mean': X_c[:, i].mean(), 'var': X_c[:, i].var()}

    def __single_prediction(self, sample):
        posteriors = {}

        # Compute posteriors for each class
        for c in self.classes:
            posterior = self.priors[c]
            for i, value in enumerate(sample):
                likelihood = self.__likelihood(self.params[c][i]['mean'], self.params[c][i]['var'], value)
                posterior *= likelihood
            posteriors[c] = posterior

        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        predictions = np.array([self.__single_prediction(sample) for sample in X])
        return predictions

    @staticmethod
    def __likelihood(mean, var, x):
        eps = 1e-10  # Added in denominator to prevent division by zero
        return (1.0 / (var * math.sqrt(2.0 * math.pi) + eps)) * math.exp(-.5 * math.pow((x - mean) / (var + eps), 2))
