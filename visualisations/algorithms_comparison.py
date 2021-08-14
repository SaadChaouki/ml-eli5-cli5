from supervised.classification.decisionTreeClassifier import DecisionTreeClassifier
from supervised.classification.naiveBayes import NaiveBayesClassifier
from supervised.classification.randomForestClassifier import RandomForestClassifier
from supervised.classification.gradientBoostingClassifier import GradientBoostingClassifier
from supervised.classification.logisticRegression import LogisticRegression
from supervised.classification.KNearestNeighbor import KNN
from supervised.classification.adaBoost import AdaBoost

from color_palette import two_colors, two_colors_map

from sklearn.naive_bayes import GaussianNB

from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("TkAgg")

if __name__ == '__main__':
    # Parameters
    grid_detail = .3
    data_size = 100

    # Create the dataset
    datasets = {
        'linear_2': make_classification(n_features=2, n_redundant=0, n_informative=2),
        'linear': make_classification(n_features=2, n_redundant=0, n_informative=2),
        'moons': make_moons(noise=0.3, random_state=0),
        'circles': make_circles(noise=0.2, factor=0.5, random_state=1)
    }

    # Create train and test sets
    for data_name in datasets:
        X, y = datasets[data_name]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        datasets[data_name] = {
            'original': (X, y),
            'train': (X_train, y_train),
            'test': (X_test, y_test)
        }

    # Create the models
    models = {
        'K-Nearest Neighbor': KNN(k=5),
        'Decision Tree': DecisionTreeClassifier(max_depth=5),
        'Random Forest': RandomForestClassifier(num_estimators=5, max_depth=5, max_features=2),
        'Logistic Regression': LogisticRegression(learning_rate=.01, iterations=5000)
    }

    # Start the pyplot grid
    figure, axes = plt.subplots(len(datasets), len(models) + 1, figsize=(27, 9))

    # Plot the input data
    for i, data_name in enumerate(datasets):
        # Set the title
        if i == 0:
            axes[i][0].set_title('Input Data')

        # Extract the data from the dictionary
        X, y = datasets[data_name]['original']

        # Plot the data
        axes[i][0].scatter(X[:, 0], X[:, 1], c=y, cmap=two_colors_map, edgecolors='k')

        # Remove the axes
        axes[i][0].set_xticks(())
        axes[i][0].set_yticks(())

    # Process the models
    for i, model_name in enumerate(models):
        for j, data_name in enumerate(datasets):
            X, y = datasets[data_name]['original']

            # Create the mesh
            x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
            y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_detail), np.arange(y_min, y_max, grid_detail))

            # Set the title
            if j == 0:
                axes[j][i + 1].set_title(model_name)

            # Plot the data

            axes[j][i + 1].scatter(X[:, 0], X[:, 1], c=y, cmap=two_colors_map, edgecolors='k')

            # Training data
            X_train, y_train = datasets[data_name]['original']

            # Train the model
            models[model_name].fit(X_train, y_train)

            # Predict on the mesh
            Z = np.array(models[model_name].predict(np.c_[xx.ravel(), yy.ravel()]))
            Z = Z.reshape(xx.shape)
            axes[j][i + 1].contourf(xx, yy, Z, cmap=two_colors_map, alpha=.5)

            # Remove the axes
            axes[j][i + 1].set_xticks(())
            axes[j][i + 1].set_yticks(())

    # Show the plot
    plt.show()
