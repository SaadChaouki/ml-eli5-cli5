from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy import stats
from scipy.optimize import minimize
from scipy.special import factorial


class RegressionResults(object):
    def __init__(self, has_constant: bool, model_type: str, x: np.ndarray, **kwargs):
        self.args: Dict[str, Any] = kwargs
        self.has_constant: bool = has_constant
        self.model_type: str = model_type

        # Creating the summary table
        self.summary_table: pd.DataFrame = pd.DataFrame(columns=['feature', *kwargs.keys()])
        if self.has_constant:
            self.summary_table['feature'] = ["constant", *[f"x{i + 1}" for i in range(0, x.shape[1] - 1)]]
        else:
            self.summary_table['feature'] = [f"x{i + 1}" for i in range(0, x.shape[1])]
        for key in kwargs.keys():
            self.summary_table[key] = kwargs[key]

    def summary(self):
        print("*" * 75)
        print("Regression Results:")
        print(f"Model Type: {self.model_type}.")
        print("*" * 75)
        print(self.summary_table)
        print("*" * 75)


class OLS(object):

    def __init__(self, x: np.ndarray, y: np.ndarray, constant: bool = True, use_t: bool = True, alpha: float = 0.05):
        self.beta: Optional[np.ndarray] = None
        self.x: np.ndarray = np.hstack((np.ones((x.shape[0], 1)), x)) if constant else x

        self.df: int = self.x.shape[0] - self.x.shape[1]
        self.y: np.ndarray = y
        self.use_t: bool = use_t

        self.alpha: float = alpha
        self.has_constant: bool = constant

    def fit(self):
        self.beta: np.ndarray = inv(self.x.T @ self.x) @ (self.x.T @ self.y)
        return self.__build_results()

    def __build_results(self) -> RegressionResults:
        fitted_values: np.ndarray = self.x @ self.beta
        residuals: np.ndarray = self.y - fitted_values

        residuals_variance: float = (residuals ** 2).sum() / self.df
        bse: np.ndarray = np.sqrt((residuals_variance * np.linalg.inv(self.x.T @ self.x)).diagonal())

        stats_model = self.beta / bse

        p_values = (stats.t.sf(abs(stats_model), df=self.df) if self.use_t else stats.norm.sf(abs(stats_model))) * 2

        q = stats.t.ppf(1 - self.alpha / 2, self.df) if self.use_t else stats.norm.ppf(1 - self.alpha / 2)
        lower = self.beta - q * bse
        upper = self.beta + q * bse

        results: RegressionResults = RegressionResults(
            self.has_constant,
            self.__class__.__name__,
            self.x,
            coef=self.beta,
            bse=bse,
            stats=stats_model,
            p_values=p_values,
            lb=lower,
            ub=upper
        )
        return results


class MaximumLikelihood(object):
    def __init__(self, x: np.ndarray, y: np.ndarray, optimization_method: str = 'l-bfgs-b'):
        self.x = x
        self.y = y
        self.method = optimization_method
        self.model = None
        self.beta = None

    def fit(self):
        initial_guesses: np.ndarray = np.array([0.01 for _ in range(0, self.x.shape[1] + 1)])
        self.model = minimize(self.mle_function, initial_guesses, method=self.method)
        self.beta = self.model['x'][:len(self.model['x']) - 1]

    def mle_function(self, parameters: np.ndarray) -> float:
        pass


class LinearMLE(MaximumLikelihood):
    def mle_function(self, parameters: np.ndarray) -> float:
        betas, stddev = parameters[:len(parameters) - 1], parameters[-1]
        y_hat = np.exp(self.x @ betas)
        return - np.sum(self.y * np.log(y_hat) - y_hat - np.log(factorial(self.y)))


class PoissonRegression(MaximumLikelihood):
    def mle_function(self, parameters: np.ndarray) -> float:
        betas, stddev = parameters[:len(parameters) - 1], parameters[-1]
        y_hat = np.exp(self.x @ betas)
        return - np.sum(self.y * np.log(y_hat) - y_hat - np.log(factorial(self.y)))
