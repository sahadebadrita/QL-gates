import igraph as ig
import numpy as np

#Scripts for building the quantum like resources and Cartesian Product

def qldit(n, k, l, d, coupling, periodic, full):
    """
    Efficiently generates the adjacency matrix for a QL-dit system.

    Parameters:
    n : int - Number of vertices per subgraph.
    k : int - Degree of regular graphs for diagonal blocks.
    l : int - Degree of coupling graphs for off-diagonal blocks.
    d : int - Number of subgraphs - 1 (determines coupling layers).
    coupling : boolean - True: coupling ON, False: coupling OFF
    periodic : boolean - True: periodic chain ON, False: periodic chain OFF
    full : boolean - True: full coupling ON, False: full coupling OFF
    Returns:
    system_matrix : np.ndarray - The full system adjacency matrix.
    """
    total_nodes = n * (d + 1)
    system_matrix = np.zeros((total_nodes, total_nodes), dtype=np.int8)  # Efficient memory allocation

    coupling_matrices = []  # Ensure coupling_matrices is always defined

    # Generate adjacency matrices for k-regular graphs (diagonal blocks)
    adjacency_matrices = [np.array(ig.Graph.K_Regular(n, k).get_adjacency().data, dtype=np.int8) for _ in range(d+1)]
    # Assign adjacency matrices to diagonal blocks
    for i in range(d + 1):
        start, end = i * n, (i + 1) * n
        system_matrix[start:end, start:end] = adjacency_matrices[i]

    if coupling == True:
        # Generate coupling adjacency matrices (l-regular graphs for off-diagonal blocks)
        coupling_matrices = [np.array(ig.Graph.K_Regular(n, l).get_adjacency().data, dtype=np.int8) for _ in range(d)]
        # Assign coupling matrices to superdiagonal and subdiagonal
        for i in range(d):
            start, end = i * n, (i + 1) * n
            start_next, end_next = (i + 1) * n, (i + 2) * n

            system_matrix[start:end, start_next:end_next] = coupling_matrices[i]  # Superdiagonal
            system_matrix[start_next:end_next, start:end] = coupling_matrices[i]  # Subdiagonal (symmetric)
    if periodic == True:
        coupling_matrices.append(np.array(ig.Graph.K_Regular(n, l).get_adjacency().data, dtype=np.int8))
        # Assign coupling matrices to the corners of R
        system_matrix[:n,-n:] = coupling_matrices[-1]  # Superdiagonal
        system_matrix[-n:,:n] = coupling_matrices[-1]  # Subdiagonal (symmetric)
    # Implement all-to-all coupling between subgraphs
    if full == True:
        # Generate a single l-regular coupling matrix
        all_to_all_coupling = np.array(ig.Graph.K_Regular(n, l).get_adjacency().data, dtype=np.int8)

        # Assign coupling matrices to all subgraph pairs
        for i in range(d + 1):
            for j in range(i + 1, d + 1):  # Avoid diagonal blocks
                start_i, end_i = i * n, (i + 1) * n
                start_j, end_j = j * n, (j + 1) * n

                system_matrix[start_i:end_i, start_j:end_j] = all_to_all_coupling
                system_matrix[start_j:end_j, start_i:end_i] = all_to_all_coupling  # Ensure symmetry
    return system_matrix


def cart_qldit(adj_mat1, adj_mat2):
    """
    Creates the cartesian product of two QL-dits.

    Parameters:
    adj_mat1 : adjacency matrix for QL-dit 1
    adj_mat2 : adjacency matrix for QL-dit 2

    Returns:
    adj_matrix_cart_kron : np.ndarray - The adjacency matrix of the cartesian product of the two QL-dits.
    """
    adj_matrix_cart_kron = np.kron(np.eye(adj_mat1.shape[0]),adj_mat2) + np.kron(adj_mat1,np.eye(adj_mat2.shape[0]))
    return adj_matrix_cart_kron

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
