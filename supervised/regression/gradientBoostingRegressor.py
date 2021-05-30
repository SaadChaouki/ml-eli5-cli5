from deep_learning.activations import Linear
from deep_learning.loss import MSELoss
from supervised.base.baseGradientBoosting import BaseGradientBoosting


class GradientBoostingRegressor(BaseGradientBoosting):
    def __init__(self, max_depth=2, num_estimators=100, minimum_sample_leaf=10, learning_rate=.1):
        super(GradientBoostingRegressor, self).__init__(max_depth=max_depth, num_estimators=num_estimators,
                                                        minimum_sample_leaf=minimum_sample_leaf,
                                                        learning_rate=learning_rate)
        self.loss = MSELoss()
        self.transformation = Linear()
