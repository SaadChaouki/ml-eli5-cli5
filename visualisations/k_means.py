from visualisations.color_palette import three_colors, three_color_map

from unsupervised.kmeans import KMeans
from sklearn.datasets import make_blobs
from matplotlib.animation import FuncAnimation
import matplotlib
import numpy as np

import argparse
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


# Create the update function for the graph
def update(i):
    plt.clf()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    fig.suptitle('K-Means Clustering', fontsize=20)
    plt.title(f'Iteration: {i + 1}')
    plt.scatter(X[:, 0], X[:, 1], c=[three_colors[cluster] for cluster in y])
    plt.imshow(predicted_area[i], interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=three_color_map, aspect='auto', origin='lower', alpha=.4)


if __name__ == '__main__':

    # Argument parsing.
    parser = argparse.ArgumentParser(description='Visualise a customer Linear Regression model in training.')
    parser.add_argument('--max_iter', type=int, help='Maximum number of iterations.', default=100)
    parser.add_argument('--center', type=int, help='Number of data centers.', default=3)
    parser.add_argument('--random_state', type=int, help='Random state for data generation.', default=42)
    parser.add_argument('--n_samples', type=int, help='Number of data points.', default=5000)
    args = parser.parse_args()

    # Setting parameters
    max_iterations = args.max_iter
    n_centers = args.center
    n_samples = args.n_samples
    random_state = args.random_state

    # Create the clusters
    X, y = make_blobs(n_samples=n_samples, centers=n_centers, n_features=2, random_state=random_state, cluster_std=1.5)

    # Clustering
    kmeans = KMeans(k=n_centers, iterations=max_iterations, random_state=random_state, track_history=True)
    kmeans.fit(X)

    # Extract centroids
    centroids = kmeans.history_centroids

    # Create decision boundary data
    h = .1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    area_data = np.c_[xx.ravel(), yy.ravel()]

    # Prepare predictions
    predicted_labels = []
    predicted_area = []
    for iteration in range(max_iterations):
        kmeans.centroids = centroids[iteration]
        area = np.array(kmeans.predict(area_data))
        area = area.reshape(xx.shape)
        predicted_labels.append(kmeans.predict(X))
        predicted_area.append(area)

    # Plotting and showing the animation.
    fig, ax = plt.subplots(figsize=(15, 6), dpi=80)
    animation = FuncAnimation(fig, update, frames=max_iterations, interval=800, repeat=False)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
