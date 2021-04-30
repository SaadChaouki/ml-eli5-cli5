import numpy as np
import math


# TODO: Investigate low performance

class Stump(object):
    def __init(self, featureIndex: int = None, featureThreshold: float = None):
        self.featureIndex = featureIndex
        self.featureThreshold = featureThreshold

    def fit(self, X, y, weights):
        bestError = float('inf')
        numberSamples = X.shape[0]
        for idx in range(X.shape[1]):
            feature = X[:, idx]
            for threshold in np.unique(feature):
                predictions = np.ones(numberSamples)
                predictions[feature < threshold] = -1

                # Compute error
                error = weights[y != predictions].sum()

                # Check the error and polarity
                error = error if error <= 0.5 else 1 - error
                polarity = 1 if error <= 0.5 else -1

                # Record best split
                if error < bestError:
                    bestError = error
                    self.featureIndex = idx
                    self.featureThreshold = threshold
                    self.polarity = polarity

        eps = 1e-10
        self.alpha = 0.5 * math.log((1 - bestError + eps) / (bestError + eps))
        predictions = [self.singlePrediction(np.array([x])) for x in X]

        newWeights = weights * np.exp(- bestError * y * predictions)
        newWeights /= sum(newWeights)

        return self, newWeights

    def singlePrediction(self, x):
        return -self.polarity if x[:, self.featureIndex][0] < self.featureThreshold else self.polarity


class AdaBoost(object):
    def __init__(self, nEstimators: int = 100):
        self.nEstimators = nEstimators
        self.stumps = []

    def fit(self, X, y):
        # Update the predictions to -1 and 1 instead of 1 and 0
        y = self.processOutput(y)

        # Initialise observations weights. The starting point is that all observations are equally important.
        numberSamples, numberFeatures = X.shape
        weights = np.ones(numberSamples) / numberSamples

        for _ in range(self.nEstimators):
            stump, weights = Stump().fit(X, y, weights)
            self.stumps.append(stump)

    def predict(self, X):
        predictions = []
        for observation in X:
            prediction = np.sign(
                sum([clf.singlePrediction(np.array([observation])) * clf.alpha for clf in self.stumps]))
            predictions.append(int(prediction))
        predictions = np.array(predictions)
        predictions[predictions < 0] = 0
        return predictions

    def processOutput(self, y):
        return y * 2 - 1


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    import matplotlib
    import numpy as np
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from sklearn.ensemble import AdaBoostClassifier

    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=10)

    # plt.scatter(X[:, 0], X[:, 1], c = y)
    # plt.show()

    clf = AdaBoost(nEstimators=4)
    clf.fit(X, y)
    predicted = clf.predict(X)

    clf2 = AdaBoostClassifier(n_estimators=4)
    clf2.fit(X, y)
    predicted2 = clf2.predict(X)


    print(sum(predicted == y)/len(y))
    print(sum(predicted2 == y) / len(y))