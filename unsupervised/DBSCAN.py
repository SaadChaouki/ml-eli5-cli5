import numpy as np
from utils.distance_measures import manhattan_distance, euclidean_distance


class DBSCAN(object):
    def __init__(self, epsilon=.5, min_neighborhood=5, distance_function=euclidean_distance):
        self.eps = epsilon
        self.min_neighborhood = min_neighborhood
        self.distance_function = distance_function
        self.visited_points = []
        self.X = None
        self.__clusters = []
        self.__noise_value = -1
        self.labels_ = np.array([])

    def __find_neighbors(self, i):
        distances = self.distance_function(self.X[i], self.X)
        distances[i] = np.inf
        return set(np.where(distances <= self.eps)[0])

    def __create_cluster(self, i, neighbors):
        # Start the cluster with the original core.
        cluster = [i]

        # Loop as long as there are more neighbors to check.
        # An alternative to this is recursion.
        while neighbors:
            # Pop a neighbor
            selected_neighbor = neighbors.pop()

            # Check if the point has already been checked
            if selected_neighbor in self.visited_points:
                continue

            # Get neighbors of the new point
            new_neighbors = self.__find_neighbors(selected_neighbor)

            # Add to visited
            self.visited_points.append(selected_neighbor)

            # Check point type. If the point is a core then get the neighbors
            # If the point has less neighbors than the min_sample, then the point is a border
            if len(new_neighbors) >= self.min_neighborhood:
                cluster.append(selected_neighbor)
                neighbors.update(new_neighbors)
            else:
                cluster.append(selected_neighbor)
                cluster.extend(new_neighbors)
                self.visited_points.extend(new_neighbors)

        return cluster

    def __create_labels(self):
        self.labels_ = np.full(self.X.shape[0], self.__noise_value)
        for cluster_index, cluster_points in enumerate(self.__clusters):
            for point in cluster_points:
                self.labels_[point] = cluster_index

    def fit_predict(self, X):
        # Store data to make it easy to access. Assign first cluster
        self.X = X

        # Loop through each point
        for i, _ in enumerate(X):

            # If the point has already been visited, move to the next point
            if i in self.visited_points:
                continue

            # Flag the point as visited
            self.visited_points.append(i)

            # Get the neighbors of the point
            neighbors = self.__find_neighbors(i)

            # Check if the point is a core
            if len(neighbors) >= self.min_neighborhood:
                cluster_data = self.__create_cluster(i, neighbors)
                self.__clusters.append(cluster_data)

        # Create the labels
        self.__create_labels()

        return self.labels_
