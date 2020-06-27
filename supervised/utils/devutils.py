import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

def generateClassificationData(n = 10):
    features, output = make_classification(n_samples = n, n_features = 2,
                                n_informative = 2, n_redundant = 0,
                                n_classes = 2)
    generatedData = pd.DataFrame(features, columns = ['var_1', 'var_2'])
    generatedData['target'] = output
    return generatedData

def plot2d(data):
    plt.clf()
    plt.title('Scatter 2D')
    plt.scatter(x = data['var_1'], y = data['var_2'], c = data['target'])
    plt.show()

if __name__ == '__main__':
    data = generateClassificationData(1000)
    plot2d(data)
