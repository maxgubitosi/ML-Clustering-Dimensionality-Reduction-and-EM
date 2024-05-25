import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import multivariate_normal


def plot_sample(data, i=None):
    if i is None:
        i = np.random.randint(data.shape[0])
    sample = data.iloc[i].drop('label').values.reshape(28, 28)
    plt.imshow(sample, cmap='gray')
    plt.show()


class Scaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0)

    def transform(self, data):
        return (data - self.mean) / self.std

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        return data * self.std + self.mean


class PCA:
    def __init__(self, n_components=2, verbose=0):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.verbose = verbose

    def fit(self, X):
        self.mean = np.mean(X, axis=0)                                      
        X_centered = X - self.mean                                          # Mean center the data
        covariance_matrix = np.cov(X_centered, rowvar=False)                # Compute covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)       # Calculate eigenvalues and eigenvectors
        idx = np.argsort(eigenvalues)[::-1]                                 # Sort eigenvectors by descending eigenvalues
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        self.components = eigenvectors[:, :self.n_components]               # Save the first n eigenvectors
        self.explained_variance = eigenvalues[:self.n_components]

    def transform(self, X):
        # Transform data to the new subspace
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def fit_transform(self, X):
        self.fit(X)
        if self.verbose != 0:
            print(f"Explained variance: {self.explained_variance}")
            print("dimensionality reduced from {} to {}".format(X.shape[1], self.n_components))
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        result = np.dot(X_transformed, self.components.T) + self.mean
        if self.verbose != 0:
            print("dimensionality increased from {} to {}".format(X_transformed.shape[1], self.components.shape[1]))
        return result

    def compute_mse(self, X_original, X_reconstructed):
        return np.mean((X_original - X_reconstructed) ** 2)