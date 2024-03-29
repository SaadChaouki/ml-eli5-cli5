import numpy as np
import random


class KMeans(object):
    def __init__(self, k, iterations=100, random_state=None, track_history=False):
        self.k = k
        self.maxIterations = iterations
        self.track_history = track_history
        self.history_centroids = []
        if random_state:
            np.random.seed(random_state)
            random.seed(random_state)

    def __generate_clusters(self, x):
        self.centroids = {i: centroid for i, centroid in enumerate(random.choices(x, k=self.k))}

    def __assign_cluster(self, x):
        distances = {i: np.linalg.norm(x - self.centroids[i]) for i in self.centroids.keys()}
        return min(distances, key=distances.get)

    def __update_clusters(self, assignments):
        self.centroids = {i: np.average(assignments[i], axis=0) for i in assignments.keys()}

    def fit(self, x):
        self.__generate_clusters(x)
        for _ in range(self.maxIterations):
            assignments = {i: [] for i in range(self.k)}
            for observation in x:
                assignments[self.__assign_cluster(observation)].append(observation)
            if self.track_history:
                self.history_centroids.append(self.centroids)
            self.__update_clusters(assignments)

    def predict(self, x):
        return [self.__assign_cluster(data) for data in x]
