import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.stats import multivariate_normal

# imports for VAE in PyTorch
import torch
from torch.utils.data import DataLoader, TensorDataset
# from torchvision import transforms
from torch import nn, optim
from torch.nn import functional as F

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
    


def plot_reconstructions(df, X_re, n_samples=10):
    fig, axes = plt.subplots(nrows=2, ncols=n_samples, figsize=(20, 4))  # 10 columns
    for i in range(10):
        sample = df.iloc[i].drop('label').values.reshape(28, 28)
        reconstructed = X_re.iloc[i].values.reshape(28, 28)
        # original image
        ax = axes[0, i]
        ax.imshow(sample, cmap='gray')
        ax.set_title('Original')
        ax.axis('off')
        # reconstruced image
        ax = axes[1, i]
        ax.imshow(reconstructed, cmap='gray')
        ax.set_title('Reconstructed')
        ax.axis('off')
    plt.tight_layout()
    plt.show()



class VAE(nn.Module):
    def __init__(self, encoder_layers, decoder_layers):
        super(VAE, self).__init__()

        # Crear el encoder
        self.encoder = nn.ModuleList()
        for in_features, out_features in encoder_layers[:-1]:
            self.encoder.append(nn.Linear(in_features, out_features))
            self.encoder.append(nn.ReLU())
        
        # Última capa del encoder sin activación ReLU
        self.fc_mu = nn.Linear(*encoder_layers[-1])
        self.fc_logvar = nn.Linear(*encoder_layers[-1])

        # Crear el decoder
        self.decoder = nn.ModuleList()
        for in_features, out_features in decoder_layers[:-1]:
            self.decoder.append(nn.Linear(in_features, out_features))
            self.decoder.append(nn.ReLU())
        
        # Última capa del decoder con activación sigmoid
        self.decoder.append(nn.Linear(*decoder_layers[-1]))
        self.decoder.append(nn.Sigmoid())

    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        for layer in self.decoder:
            z = layer(z)
        return z

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD
    
    def train_model(self, train_loader, device, epochs=10, lr=1e-3, verbose=1):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.to(device)
        for epoch in range(epochs):
            self.train()
            train_loss = 0
            for batch_idx, (data,) in enumerate(train_loader):
                data = data.to(device)
                optimizer.zero_grad()
                recon_batch, mu, logvar = self(data)
                loss = self.loss_function(recon_batch, data, mu, logvar)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()
            if verbose != 0:
                print(f'Epoch: {epoch}, Loss: {train_loss / len(train_loader.dataset)}')