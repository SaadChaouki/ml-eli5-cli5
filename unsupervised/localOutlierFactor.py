import numpy as np


def manhattan_distance(point, X):
    return np.sum(np.abs(point - X), axis=1, dtype=np.float)


def euclidean_distance(point, X):
    return np.sqrt(np.sum(np.square(point - X), axis=1))


class LocalOutlierFactor(object):
    def __init__(self, k_neighbors=20, distance_function=euclidean_distance):
        self.k_neighbors = k_neighbors
        self.distance_function = distance_function

    def __get_k_neighbors(self, i, point, X):
        distances = self.distance_function(point, X)
        distances[i] = np.inf
        neighbors = distances.argsort()[:self.k_neighbors]
        return {'neighborhood': neighbors, 'distances': distances[neighbors]}

    def __compute_reach_distance(self, point_index, neighbors):
        zipped_neighbors = zip(neighbors[point_index]['neighborhood'], neighbors[point_index]['distances'])
        total_distance = 0
        for neighbor in zipped_neighbors:
            distance_points = neighbor[1]
            distance_k_th = neighbors[neighbor[0]]['distances'][self.k_neighbors - 1]
            total_distance = total_distance + max(distance_points, distance_k_th)
        return total_distance

    @staticmethod
    def __compute_local_reachability_density(i, neighbors, reach_distances):
        return len(neighbors[i]['neighborhood']) / reach_distances[i]

    @staticmethod
    def __compute_local_outlier_factor(i, neighbors, lrd, reach_distance):
        point_neighbors = neighbors[i]['neighborhood']
        total_lrd = sum([lrd[k] for k in point_neighbors])
        return (total_lrd * reach_distance[i]) / (len(point_neighbors) ** 2)

    def fit_predict(self, X):
        neighbors = {i: self.__get_k_neighbors(i, point, X) for i, point in enumerate(X)}
        reach_distances = {i: self.__compute_reach_distance(i, neighbors) for i in neighbors.keys()}
        lrd = {i: self.__compute_local_reachability_density(i, neighbors, reach_distances) for i in neighbors.keys()}
        lof = {i: self.__compute_local_outlier_factor(i, neighbors, lrd, reach_distances) for i in neighbors.keys()}
        return np.array(list(lof.values()))
