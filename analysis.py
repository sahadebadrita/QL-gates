# ============================================================
# Imports
# ============================================================
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import expm, logm
from scipy.integrate import solve_ivp
from scipy.stats import unitary_group
import igraph as ig
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
from scipy import sparse

def compute_eigen_decomposition(adj_matrix, top_k=None):
    """Compute eigenvalues and eigenvectors of a symmetric adjacency matrix.
       Returns only the top_k eigenpairs if specified."""
    eigenvalues, eigenvectors = np.linalg.eigh(adj_matrix)
    if top_k:
        indices = np.argsort(eigenvalues)[-top_k:]  # Get indices of top_k largest eigenvalues
        return eigenvalues[indices][::-1], eigenvectors[:, indices][:, ::-1]  # Return sorted
    return eigenvalues, eigenvectors

def extract_subgraph_eigenvectors(adj_matrix, n, d):
    """Compute the leading eigenvector (corresponding to the largest eigenvalue) of each subgraph."""
    subgraph_vectors = np.zeros((n, d+1))  # Each column stores a leading eigenvector of a subgraph

    for i in range(d+1):
        sub_adj = adj_matrix[i*n:(i+1)*n, i*n:(i+1)*n]  # Extract subgraph adjacency matrix
        _, eigvecs = compute_eigen_decomposition(sub_adj, top_k=1)  # Get top eigenvector
        subgraph_vectors[:, i] = eigvecs[:, 0]  # Extract as 1D array

    return subgraph_vectors

def compute_projection_matrix(full_eigenvectors, subgraph_vectors, n, d):
    """Project the top part of full eigenvectors onto the subgraph eigenvectors."""
    projection_matrix = np.zeros((d+1, d+1))  # Store projections

    for j in range(d+1):
        for i in range(d+1):
            # Extract the corresponding n-length portion of the full eigenvectors
            full_segment = full_eigenvectors[i*n:(i+1)*n, j]  # Select the i-th portion

            # Compute projection onto the corresponding subgraph eigenvector
            projection_matrix[i, j] = full_segment @ subgraph_vectors[:, i]  # Dot product projection
        print(j,np.linalg.norm(projection_matrix[:,j]))
    return projection_matrix
