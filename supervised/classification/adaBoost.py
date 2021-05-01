import numpy as np
import math

# TODO: Speed up predictions.

class Stump(object):
    def __init__(self):
        self.feature_index = None
        self.feature_threshold = None
        self.polarity = 1
        self.alpha = None

    def fit(self, X, y, weights):
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
        self.alpha = 0.5 * math.log((1 - best_split_error) / (best_split_error + 1e-10))

        # Predictions
        predictions = [self.single_prediction(np.array([x])) for x in X]

        # Update the weights
        updated_weights = weights * np.exp(-self.alpha * y * predictions)
        updated_weights /= sum(updated_weights)

        return self, updated_weights

    def single_prediction(self, x):
        return -self.polarity if x[:, self.feature_index][0] < self.feature_threshold else self.polarity


class AdaBoost(object):
    def __init__(self, num_estimators: int = 100):
        self.num_estimators = num_estimators
        self.stumps = []

    def fit(self, X, y):

        # Update the predictions to -1 and 1 instead of 1 and 0
        y = self.process_output(y)

        # Initialise observations weights. The starting point is that all observations are equally important.
        num_samples, num_features = X.shape
        weights = np.ones(num_samples) / num_samples

        for _ in range(self.num_estimators):
            stump, weights = Stump().fit(X, y, weights)
            self.stumps.append(stump)

    def predict(self, X):
        predictions = []
        for observation in X:
            prediction = np.sign(
                sum([clf.single_prediction(np.array([observation])) * clf.alpha for clf in self.stumps]))
            predictions.append(int(prediction))
        predictions = np.array(predictions)
        predictions[predictions < 0] = 0
        return predictions

    @staticmethod
    def process_output(y):
        return y * 2 - 1