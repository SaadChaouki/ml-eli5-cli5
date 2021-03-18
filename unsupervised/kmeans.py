import numpy as np
import random


class KMeans():
    def __init__(self, k, iterations=100):
        self.k = k
        self.maxIterations = iterations

    def __generateClusters(self, x):
        self.centroids = {i: centroid for i, centroid in enumerate(random.choices(x, k=self.k))}

    def __assignCluster(self, x):
        distances = {i: np.linalg.norm(x - self.centroids[i]) for i in self.centroids.keys()}
        return min(distances, key=distances.get)

    def __updateClusters(self, assignments):
        self.centroids = {i: np.average(assignments[i], axis=0) for i in assignments.keys()}
    
    def fit(self, x):
        self.__generateClusters(x)
        for _ in range(self.maxIterations):
            assignments = {i: [] for i in range(self.k)}
            for observation in x:
                assignments[self.__assignCluster(observation)].append(observation)
            self.__updateClusters(assignments)

    def predict(self, x):
        return [self.__assignCluster(data) for data in x]
