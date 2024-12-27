# Clustering, Dimensionality Reduction, and EM Algorithm

This repository contains the solutions to the fourth practical assignment for the course “I302 - Machine Learning and Deep Learning” for the last semester of 2023.

## Table of Contents
- [Introduction](#introduction)
- [Problem Descriptions](#problem-descriptions)
  - [1. Data Clustering](#1-data-clustering)
  - [2. Dimensionality Reduction](#2-dimensionality-reduction)
  - [3. Expectation-Maximization](#3-expectation-maximization)
- [Results](#results)
  - [Data Clustering Results](#data-clustering-results)
  - [Dimensionality Reduction Results](#dimensionality-reduction-results)
  - [Expectation-Maximization Results](#expectation-maximization-results)

## Introduction

This repository contains a collection of Jupyter Notebooks and Python scripts developed for the fourth practical assignment of the “I302 - Machine Learning and Deep Learning” course. The assignment focuses on three key topics: Data Clustering, Dimensionality Reduction, and Expectation-Maximization (EM).

The problems were solved using basic tools such as NumPy, Pandas, Matplotlib, and optionally PyTorch for more advanced tasks. Each problem is organized into separate folders, with corresponding notebooks and auxiliary scripts when necessary.

## Problem Descriptions

1. Data Clustering

The goal of this problem is to analyze the dataset clustering.csv using various clustering techniques:
	•	K-Means Algorithm:
	•	Implemented to cluster data points.
	•	The optimal number of clusters was determined using the “elbow method”.
	•	Gaussian Mixture Model (GMM):
	•	Implemented to fit a probabilistic model to the data.
	•	Initialization was performed using K-Means.
	•	DBSCAN Algorithm:
	•	Explored the effect of varying parameters such as ϵ (neighborhood radius) and K (minimum points per dense region).
	•	Optimal parameters were selected, and clusters were visualized.

The implementation can be found in:
	•	Problema 1/
	•	Problema_1.ipynb
	•	funciones_1.py

2. Dimensionality Reduction

This problem uses the MNIST dataset.csv to reduce the dimensionality of image data:
	•	Principal Component Analysis (PCA):
	•	Implemented to reduce data dimensionality.
	•	The reconstruction error was analyzed across varying numbers of principal components.
	•	Image Reconstruction:
	•	Original and reconstructed images were compared visually.
	•	(Optional) Variational Autoencoder (VAE):
	•	Built using PyTorch to compare its performance with PCA.

The implementation can be found in:
	•	Problema 2/
	•	Problema_2.ipynb
	•	funciones_2.py

3. Expectation-Maximization

This problem focuses on deriving and implementing the Expectation-Maximization (EM) algorithm for Gaussian Mixture Models (GMM):
	•	E-step:
	•	Derivation of the Q(w, w₀) function.
	•	M-step:
	•	Optimization of parameters µ, Σ, and π.
	•	Mathematical Demonstration:
	•	Proof that the Q(w, w₀) function is a lower bound on the log-likelihood.

The implementation can be found in:
	•	Problema 3/
	•	(Mathematical derivations are included as an image inside the notebook.)

## Results

Clustering Results

Visualization of the clustering results for K-Means, GMM, and DBSCAN. Each plot shows the data distribution, assigned clusters, and centroids (if applicable).

PCA Results

Analysis of the reconstruction error as a function of the number of principal components, along with side-by-side visual comparisons of original and reconstructed images.

Expectation-Maximization Results

Mathematical derivations, optimized parameter values, and graphical representation of the Gaussian Mixture Model (GMM) results.


### Contributions

This practical assignment was developed by [Máximo Gubitosi] as part of the “I302 - Machine Learning and Deep Learning” course during the last semester of 2023.
