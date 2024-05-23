import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class KMeans:
    def __init__(self, num_clusters=2, max_iters=20):
        self.num_clusters = num_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.loss = 0

    def initialize_centroids(self, data):
        # Randomly initialize centroids by selecting random data points
        indices = np.random.choice(data.shape[0], self.num_clusters, replace=False)
        self.centroids = data[indices, :]

    def euclidian_distance(self, data):
        # Calculate Euclidean distance from each point to each centroid
        return np.sqrt(((data - self.centroids[:, np.newaxis])**2).sum(axis=2))

    def assign_clusters(self, data):
        # Assign each data point to the closest centroid
        distances = self.euclidian_distance(data)
        return np.argmin(distances, axis=0)

    def update_centroids(self, data, assignments):
        # Update centroid positions to the mean of assigned data points
        for i in range(self.num_clusters):
            if np.any(assignments == i):
                self.centroids[i] = data[assignments == i].mean(axis=0)

    def compute_loss(self, data, assignments):
        # Calculate loss as the sum of squared distances to the closest centroid
        self.loss = sum((np.linalg.norm(data[assignments == i] - self.centroids[i], axis=1)**2).sum() for i in range(self.num_clusters))
        return self.loss

    def fit(self, data):
        self.initialize_centroids(data)
        for _ in range(self.max_iters):
            assignments = self.assign_clusters(data)
            self.update_centroids(data, assignments)
        self.compute_loss(data, assignments)
        return assignments

    def plot_clusters(self, data):
        # Plot the final clusters and centroids
        assignments = self.fit(data)
        plt.figure(figsize=(12, 8))
        plt.scatter(data[:, 0], data[:, 1], c=assignments, alpha=0.5, cmap='rainbow')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color='black', marker='x')
        plt.title("Final Cluster Visualization")
        plt.xlabel("A")
        plt.ylabel("B")
        plt.show()
