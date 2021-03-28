from unsupervised.dbscan import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import matplotlib
from visualisations.color_palette import two_colors
from matplotlib.lines import Line2D

matplotlib.use("TkAgg")

if __name__ == '__main__':
    X, y = make_moons(n_samples=1000, noise=.08, random_state=42)

    # Plot
    fig, ax = plt.subplots(figsize=(15, 6), dpi=80)
    fig.suptitle('DBSCAN', fontsize=20)
    plt.title('Epsilon: .125 | Minimum Neighborhood: 5')

    # DBSCAN
    model = DBSCAN(epsilon=.125, min_neighborhood=5)
    clusters = model.fit_predict(X)

    colours_dict = {
        -1: 'black',
        0: two_colors[0],
        1: two_colors[1]
    }

    # Original Data
    scatter = ax.scatter(X[:, 0], X[:, 1], c=[colours_dict[x] for x in clusters])
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor=two_colors[0]),
                       Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor=two_colors[1]),
                       Line2D([0], [0], marker='o', color='w', label='Scatter', markerfacecolor='black')]

    plt.legend(handles=legend_elements, labels=['Cluster 1', 'Cluster 2', 'Noise'])
    fig.savefig('animations/dbscan.png')
