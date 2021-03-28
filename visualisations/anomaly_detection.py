from unsupervised.isolationForest import IsolationForest
from unsupervised.localOutlierFactor import LocalOutlierFactor
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from visualisations.color_palette import anomaly_map
from sklearn.preprocessing import MinMaxScaler
import numpy as np

matplotlib.use("TkAgg")

if __name__ == '__main__':
    X, y = make_blobs(centers=2, n_samples=1000, n_features=2, random_state=42)

    # # Add outlier
    # X[0] = np.array([10, 10])

    # Isolation Forest
    ilf = IsolationForest()
    ilf.fit(X)
    ilf_scores = ilf.predict(X)
    ilf_scores = MinMaxScaler().fit_transform(ilf_scores.reshape(-1, 1))[:, 0]

    # Local Outlier Factor
    lof = LocalOutlierFactor()
    lof_scores = lof.fit_predict(X)
    lof_scores = MinMaxScaler().fit_transform(lof_scores.reshape(-1, 1))[:, 0]

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

    # Legend
    fig.colorbar(color_map, ax=ax.ravel().tolist(), shrink=0.95, orientation='vertical', label='Anomaly Score')
    fig.savefig('animations/anomaly_detection.png')

    plt.show()
