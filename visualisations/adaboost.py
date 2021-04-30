from supervised.classification.adaBoost import AdaBoost
from unsupervised.principalComponentAnalysis import PCA
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

if __name__ == '__main__':
    X, y = make_classification(n_samples= 1000,n_features=5, n_redundant=2, n_informative=3)
    pca = PCA(num_components=2)


    X_transformed = pca.transform(X)

    clf = AdaBoost(nEstimators=50)
    clf.fit(X, y)
    preds = clf.predict(X)

    t = AdaBoostClassifier(n_estimators=50)
    t.fit(X, y)
    pred_sk = t.predict(X)

    print(preds)

    print(sum(preds == y))
    print(sum(pred_sk == y))

    plt.scatter(X[:, 0], X[:, 1], c = y)
    plt.show()
