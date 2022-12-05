import numpy as np

from modelstatistics.regression import OLS, LinearMLE
import pandas as pd
from linearmodels.datasets import jobtraining
from linearmodels.panel import PooledOLS, PanelOLS, FirstDifferenceOLS, BetweenOLS, RandomEffects
import statsmodels.api as sm
import abc
from typing import List


# Panel Models with entity effects.

class PanelModel(object, metaclass=abc.ABCMeta):

    def __init__(self, data: pd.DataFrame, x_columns: List[str], target: str, add_constant: bool = False):
        self.raw_data: pd.DataFrame = data[x_columns + [target]]
        self.group_id: str = self.raw_data.index.names[0]
        self.x_columns: List[str] = x_columns
        self.target: str = target
        self.add_constant: bool = add_constant
        self.model = None
        self.betas = None

    @abc.abstractmethod
    def transform(self):
        pass

    def fit(self):
        X, y = self.transform()
        self.model = OLS(X, y, constant=self.add_constant).fit()
        return self.model


class OLSPooled(PanelModel):
    def transform(self):
        return self.raw_data[self.x_columns].values, self.raw_data[self.target].values


class FixedEffects(PanelModel):
    def transform(self):
        transformed_data: pd.DataFrame = self.raw_data - self.raw_data.groupby(self.group_id).transform('mean')
        return transformed_data[self.x_columns].values, transformed_data[self.target].values


class FirstDifference(PanelModel):
    def transform(self):
        transformed_data = self.raw_data.groupby(self.group_id).diff().dropna()
        return transformed_data[self.x_columns].values, transformed_data[self.target].values


class Between(PanelModel):
    def transform(self):
        transformed_data = self.raw_data.groupby(self.group_id).mean()
        return transformed_data[self.x_columns].values, transformed_data[self.target].values


class RandomEffects(PanelModel):
    def transform(self):
        pass

if __name__ == '__main__':
    data = jobtraining.load()
    data = data.set_index(["fcode", "year"]).dropna()

    X = data[['employ', 'avgsal']]
    y = data['sales']
    # Pooled OLS
    mod = BetweenOLS(y, X)
    linear_model_pooled = mod.fit()
    print(linear_model_pooled.summary)

    my_pooled_ols = Between(data, ['employ', 'avgsal'], 'sales').fit()
    print(my_pooled_ols.summary_table['coef'])


