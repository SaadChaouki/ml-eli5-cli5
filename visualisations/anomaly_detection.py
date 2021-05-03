
from unsupervised.isolationForest import IsolationForest
from unsupervised.localOutlierFactor import LocalOutlierFactor
from visualisations.color_palette import anomaly_map

import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

import argparse

matplotlib.use("TkAgg")

if __name__ == '__main__':

    # Argument parsing.
    parser = argparse.ArgumentParser(description='Visualise a custom Linear Regression model in training.')
    parser.add_argument('--random_state', type=int, help='Random state for data generation.', default=42)
    parser.add_argument('--n_samples', type=int, help='Number of data points.', default=1000)
    parser.add_argument('--centers', type=int, help='Number of centers in the data', default=2)
    args = parser.parse_args()

    # Generate data
    X, y = make_blobs(centers=args.centers, n_samples=args.n_samples, n_features=2, random_state=args.random_state)

    # Isolation Forest
    ilf = IsolationForest()
    ilf.fit(X)
    ilf_scores = ilf.predict(X)
    ilf_scores = MinMaxScaler().fit_transform(ilf_scores.reshape(-1, 1)).ravel()

    # Local Outlier Factor
    lof = LocalOutlierFactor()
    lof_scores = lof.fit_predict(X)
    lof_scores = MinMaxScaler().fit_transform(lof_scores.reshape(-1, 1)).ravel()

    # Plot
    fig, ax = plt.subplots(figsize=(22, 9), dpi=80, nrows=1, ncols=2)
    fig.suptitle('Anomaly Detection', fontsize=20)

    # Plot isolation forest scores
    ax[0].set_title('Isolation Forest')
    color_map = ax[0].scatter(X[:, 0], X[:, 1], c=ilf_scores, cmap=anomaly_map)
    ax[0].set_xlabel('Feature 1')
    ax[0].set_ylabel('Feature 2')

    # Plot local outlier factor
    ax[1].set_title('Local Outlier Factor')
    ax[1].scatter(X[:, 0], X[:, 1], c=lof_scores, cmap=anomaly_map)
    ax[1].set_xlabel('Feature 1')
    ax[1].set_ylabel('Feature 2')

    # Legend and showing plot.
    fig.colorbar(color_map, ax=ax.ravel().tolist(), shrink=0.95, orientation='vertical', label='Anomaly Score')
    plt.show()
