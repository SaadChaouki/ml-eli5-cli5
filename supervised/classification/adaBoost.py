import numpy as np
import math


class Stump(object):
    def __init__(self):
        self.feature_index = None
        self.feature_threshold = None
        self.polarity = 1
        self.alpha = None

    def fit(self, X, y, weights):
        # Set the best split error
        best_split_error = float('inf')
        n_samples = X.shape[0]

        for idx in range(X.shape[1]):
            feature = X[:, idx]

            for threshold in np.unique(feature):
                polarity = 1
                predictions = np.ones(n_samples)
                predictions[feature < threshold] = -1

                # Compute error
                error = weights[y != predictions].sum()

                # Check the error and polarity
                if error > .5:
                    error = 1 - error
                    polarity = -1

                # Record best split
                if error < best_split_error:
                    best_split_error = error
                    self.feature_index = idx
                    self.feature_threshold = threshold
                    self.polarity = polarity

        # Compute the alpha
        self.alpha = .5 * math.log((1 - best_split_error) / (best_split_error + 1e-100))

        # Predictions
        predictions = self.predict(X)

        # Update the weights
        updated_weights = weights * np.exp(-self.alpha * y * predictions)
        updated_weights /= sum(updated_weights)

        return self, updated_weights

    def predict(self, x: np.array) -> np.array:
        predictions = np.ones(len(x))
        negative_idx = (self.polarity * x[:, self.feature_index] < self.polarity * self.feature_threshold)
        predictions[negative_idx] = -1
        return predictions


class AdaBoost(object):
    def __init__(self, num_estimators: int = 100) -> None:
        self.num_estimators = num_estimators
        self.stumps = None

    def fit(self, X: np.array, y: np.array) -> None:
        # Create array to hold the stumps
        self.stumps = []

        # Update the predictions to -1 and 1 instead of 1 and 0
        y = self.process_output(y)

        # Initialise observations weights. The starting point is that all observations are equally important.
        num_samples, num_features = X.shape
        weights = np.ones(num_samples) / num_samples

        for _ in range(self.num_estimators):
            stump, weights = Stump().fit(X, y, weights)
            self.stumps.append(stump)

    def predict(self, X: np.array) -> np.array:
        predictions = []
        for observation in X:
            prediction = np.sign(
                sum([clf.predict(np.array([observation])) * clf.alpha for clf in self.stumps]))
            predictions.append(int(prediction))
        predictions = (np.array(predictions) + 1) / 2
        return predictions

    @staticmethod
    def process_output(y: np.array) -> np.array:
        return y * 2 - 1
