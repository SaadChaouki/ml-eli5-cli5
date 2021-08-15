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
    def test(sample: np.array, population_mean: float, population_std: float) -> float:
        """
        Function to compute the Z-score
        Formula:
            Z = (x - u) / (var / sqrt(n))
            where:
                Z: Compute z score
                x: Sample Average
                u: Population Average
                var: Population Standard Deviation
                n: Sample Size
        """
        z_score = (sample.mean() - population_mean) / (population_std / np.sqrt(len(sample)))
        return z_score

    def __call__(self, sample: np.array, population_mean: float, population_std: float):
        return self.test(sample=sample,
                         population_mean=population_mean,
                         population_std=population_std)


if __name__ == '__main__':
    print('saad')
    x = np.array([0, 1, 0, 0, 0])
    s = ZTest.test(x, .3, 100)
    print(s)
    print(x.mean())
