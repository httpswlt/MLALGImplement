# coding:utf-8
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import random


def load_data():
    x, y = make_blobs(n_samples=100, n_features=2, centers=6, random_state=1234, cluster_std=0.6)
    return x, y


class KMeans(object):
    def __init__(self, x, n_clusters, max_iter):
        super(KMeans, self).__init__()
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.x = x
        self.col_min = x.min(axis=0)
        self.col_max = x.max(axis=0)
        # generate random centroids.
        self.random_pool = range(0, self.x.shape[0])
        self.__centroids = []
        for _ in range(self.n_clusters):
            self.__centroids.append(self._random_centroids())

    @property
    def centroids(self):
        return np.array(self.__centroids)

    def fit(self):
        # get the best cluster point by iteration.
        for i in range(self.max_iter):
            clusters_indx = self.pred(self.x)
            for c in range(self.n_clusters):
                if c in clusters_indx:
                    self.__centroids[c] = np.mean(self.x[c == clusters_indx], axis=0)
                else:
                    self.__centroids[c] = self._random_centroids()

    def _random_centroids(self):
        # first solution: random choose
        # random_point = self.x[random.choice(self.random_pool)]

        # Second solution: the point by selected need to have a long distance.
        if not len(self.__centroids):
            random_point = self.x[random.choice(self.random_pool)]
        else:
            distances = self.distance(self.x)
            if distances.shape[1] == 1:
                random_point = self.x[np.argmax(np.sum(distances, axis=1, keepdims=True))]
            else:
                random_point = self.x[self.calculate_best_point(distances)]
        return random_point

    def pred(self, x):
        distances = self.distance(x)
        return np.argmin(distances, axis=1)

    def distance(self, x):
        result = None
        for centroid in self.__centroids:
            temp = self.dis_l1(x, centroid)
            if isinstance(result, np.ndarray):
                result = np.hstack((result, temp))
            else:
                result = temp
        return result

    @staticmethod
    def calculate_best_point(x):
        min_ratio = float('inf')
        indx = 0
        for i, dis in enumerate(x):
            ratio = np.max(dis) / float(np.min(dis))
            if ratio < min_ratio:
                min_ratio = ratio
                indx = i
        return indx

    @staticmethod
    def dis_l2(a, b):
        return np.sqrt(np.sum((a - b) ** 2, axis=1, keepdims=True))

    @staticmethod
    def dis_l1(a, b):
        return np.sum(np.abs(a - b), axis=1, keepdims=True)


def plot(x, y, centroids, title='result'):
    plt.figure(figsize=(6, 6))
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.scatter(centroids[:, 0], centroids[:, 1], s=150, c='red')
    plt.title(title)
    # plt.show()


def main():
    n_clusters = 6
    max_iter = 10000
    x, y = load_data()
    kmeans = KMeans(x, n_clusters, max_iter)
    plot(x, y, kmeans.centroids, 'start')
    kmeans.fit()
    plot(x, y, kmeans.centroids, 'result')
    plt.show()


if __name__ == '__main__':
    main()
