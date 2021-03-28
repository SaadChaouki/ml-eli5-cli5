from unsupervised.kmeans import KMeans
from sklearn.datasets import make_blobs
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib
import numpy as np
from visualisations.color_palette import three_colors, three_color_map

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

if __name__ == '__main__':
    max_iterations = 20
    n_centers = 3

    X, y = make_blobs(n_samples=5000, centers=n_centers, n_features=2, random_state=42, cluster_std=1.5)

    kmeans = KMeans(k=n_centers, iterations=max_iterations, random_state=42, track_history=True)
    kmeans.fit(X)

    # Centroids
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

    # Setting colours. Fixing these to ensure that the colours of the area match with the colours of the points.

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

    # Plotting and saving the gif
    fig, ax = plt.subplots(figsize=(15, 6), dpi=80)
    animation = FuncAnimation(fig, update, frames=max_iterations, interval=800, repeat=False)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    animation.save('animations/k-means.gif', writer=PillowWriter(fps=60))
