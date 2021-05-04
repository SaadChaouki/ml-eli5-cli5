from supervised.regression.decisionTreeRegressor import DecisionTreeRegressor
from deep_learning.activations import Sigmoid, Linear
import numpy as np

from deep_learning.loss import MSELoss, BCELoss

class BaseGradientBoosting(object):
    def __init__(self, max_depth=2, num_estimators=100, minimum_sample_leaf=10, learning_rate=.1):
        self.max_depth = max_depth
        self.num_estimators = num_estimators
        self.minimum_sample_leaf = minimum_sample_leaf
        self.learning_rate = learning_rate
        self.models = [DecisionTreeRegressor(
            max_depth=self.max_depth,
            minimum_sample_leaf=self.minimum_sample_leaf
        ) for _ in range(self.num_estimators)]
        self.loss = BCELoss()

    def fit(self, X, y):
        # Starting with the average of y
        y_predicted = np.full(len(y), np.mean(y))

        # Fitting the models
        for model in self.models:
            gradient = self.loss.gradient(y, y_predicted)
            model.fit(X, gradient)
            gradient_prediction = model.predict(X)
            y_predicted -= self.learning_rate * np.array(gradient_prediction)

    def predict(self, X):
        y_pred = np.array([])
        # Make predictions
        for model in self.models:
            update = model.predict(X)
            update = np.multiply(self.learning_rate, update)
            y_pred = -update if not y_pred.any() else y_pred - update

        return Sigmoid()(y_pred)

if __name__ == '__main__':
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=5000, n_informative=2, n_features=2, n_redundant=0, random_state=42)

    gbt = BaseGradientBoosting(num_estimators=1)
    gbt.fit(X, y)

    # Testing
    y_preds = gbt.predict(X)
    # 4344
    # Accuracy
    print(sum(np.round(y_preds) == y))