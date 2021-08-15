import abc
import numpy as np


class StatisticalTest(object, metaclass=abc.ABCMeta):

    def __init__(self):
        pass

    @abc.abstractmethod
    def test(self, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, **kwargs):
        pass


class ZTest(StatisticalTest):

    @staticmethod
    def test(sample: np.array, population_mean: float, population_variance: float) -> float:
        """
        Function to compute the Z-score
        Formula:
            Z = (x - u) / (var / sqrt(n))
            where:


        """
        z_score = (sample.mean() - population_mean) / (population_variance / np.sqrt(len(sample)))
        return z_score

    def __call__(self, sample: np.array, population_mean: float, population_variance: float):
        return self.test(sample=sample,
                         population_mean=population_mean,
                         population_variance=population_variance)


if __name__ == '__main__':
    print('saad')
    s = ZTest.test(np.array([0, 1, 0, 0, 0]), .2, 100)


