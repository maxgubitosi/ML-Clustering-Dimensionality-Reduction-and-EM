import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import multivariate_normal



class KMeans:
    def __init__(self, num_clusters=2, max_iters=20, seed=42):
        self.num_clusters = num_clusters
        self.max_iters = max_iters
        self.centroids = None
        self.loss = 0
        self.seed = seed

    def initialize_centroids(self, data):
        # Set the random seed for reproducibility
        np.random.seed(self.seed)
        # Randomly initialize centroids by selecting random data points (with seed)
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
        plt.scatter(data[:, 0], data[:, 1], c=assignments, alpha=0.5, cmap='turbo')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], color='black', marker='x')
        plt.title(f"Final Cluster Visualization ({self.num_clusters} clusters)")
        plt.xlabel("A")
        plt.ylabel("B")
        plt.show()



class GMM:
    def __init__(self, num_components=2, max_iters=100, seed=42):
        self.num_components = num_components
        self.max_iters = max_iters
        self.means = None
        self.covariances = None
        self.weights = None
        self.seed = seed
        self.log_likelihood = 0

    def initialize_parameters_with_kmeans(self, data):
        # Set the random seed for reproducibility
        np.random.seed(self.seed)
        # Initialize using KMeans results
        kmeans = KMeans(num_clusters=self.num_components, max_iters=100, seed=self.seed)
        kmeans.fit(data)
        self.means = kmeans.centroids
        
        # Initialize covariances and weights
        self.covariances = np.zeros((self.num_components, data.shape[1], data.shape[1]))
        self.weights = np.zeros(self.num_components)
        
        assignments = kmeans.assign_clusters(data)
        for k in range(self.num_components):
            points_in_cluster = data[assignments == k]
            if len(points_in_cluster) > 1:
                self.covariances[k] = np.cov(points_in_cluster, rowvar=False)
            else:
                self.covariances[k] = np.eye(data.shape[1])  # Avoid singular matrix
            self.weights[k] = len(points_in_cluster) / len(data)

    def e_step(self, data):
        responsibilities = np.zeros((data.shape[0], self.num_components))
        for k in range(self.num_components):
            rv = multivariate_normal(self.means[k], self.covariances[k])
            responsibilities[:, k] = self.weights[k] * rv.pdf(data)
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def m_step(self, data, responsibilities):
        n_samples = data.shape[0]
        for k in range(self.num_components):
            weight_k = responsibilities[:, k].sum()
            mean_k = (data * responsibilities[:, k][:, np.newaxis]).sum(axis=0) / weight_k
            cov_k = (data - mean_k).T @ ((data - mean_k) * responsibilities[:, k][:, np.newaxis]) / weight_k
            self.means[k] = mean_k
            self.covariances[k] = cov_k
            self.weights[k] = weight_k / n_samples

    def compute_log_likelihood(self, data):
        total_likelihood = 0
        for k in range(self.num_components):
            rv = multivariate_normal(self.means[k], self.covariances[k])
            total_likelihood += self.weights[k] * rv.pdf(data)
        self.log_likelihood = np.sum(np.log(total_likelihood))

    def fit(self, data):
        # Fit the model to the data using the EM algorithm with KMeans initialization
        self.initialize_parameters_with_kmeans(data)
        for _ in range(self.max_iters):
            responsibilities = self.e_step(data)
            self.m_step(data, responsibilities)
            self.compute_log_likelihood(data)

    def plot_clusters(self, data):
        responsibilities = self.e_step(data)
        assignments = responsibilities.argmax(axis=1)
        plt.figure(figsize=(12, 8))
        plt.scatter(data[:, 0], data[:, 1], c=assignments, alpha=0.5, cmap='turbo')
        plt.scatter(self.means[:, 0], self.means[:, 1], color='black', marker='x')
        plt.title(f"Final Cluster Visualization ({self.num_components} clusters)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()



class DBSCAN:
    def __init__(self, eps=20000, min_samples=10):
        self.eps = eps
        self.min_samples = min_samples
        self.clusters = None
        self.visited = None
        self.data = None
        self.noise = -1 

    def range_query(self, point):
        # Find all points within eps distance of point
        return np.where(np.linalg.norm(self.data - point, axis=1) <= self.eps)[0]

    def expand_cluster(self, point_index, neighbors, cluster_id):
        # Expand the cluster from the point
        self.clusters[point_index] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            if not self.visited[neighbor]:
                self.visited[neighbor] = True
                new_neighbors = self.range_query(self.data[neighbor])
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.append(neighbors, new_neighbors)
            if self.clusters[neighbor] == 0:
                self.clusters[neighbor] = cluster_id
            i += 1

    def fit(self, data):
        self.data = data
        self.clusters = np.zeros(len(data), dtype=int)  # Initialize all points as cluster 0, which will represent noise
        self.visited = np.zeros(len(data), dtype=bool)
        cluster_id = 1  # Start cluster IDs from 1

        for point_index in range(len(data)):
            if not self.visited[point_index]:
                self.visited[point_index] = True
                neighbors = self.range_query(self.data[point_index])
                if len(neighbors) < self.min_samples:
                    self.clusters[point_index] = self.noise
                else:
                    self.expand_cluster(point_index, neighbors, cluster_id)
                    cluster_id += 1

    def plot_clusters(self, data, ax=None):
        self.fit(data)  # Ensure the data is fitted before plotting
        unique_clusters = np.unique(self.clusters)
        colors = plt.cm.turbo(np.linspace(0, 1, len(unique_clusters)))

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
            standalone = True
        else:
            standalone = False

        for cluster, color in zip(unique_clusters, colors):
            if cluster == self.noise:
                label = 'Noise'
            else:
                label = f'Cluster {cluster}'
            ax.scatter(data[self.clusters == cluster, 0], data[self.clusters == cluster, 1], c=[color], label=label, alpha=0.5)

        ax.set_title("DBSCAN Clustering Results" if standalone else f'eps={self.eps}, min_samples={self.min_samples}')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        # Only add the legend if it is a standalone plot
        if standalone:
            # legend outside of plot
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        if standalone:
            plt.show()

    def cluster_value_counts(self):
        return pd.Series(self.clusters).value_counts().sort_index()