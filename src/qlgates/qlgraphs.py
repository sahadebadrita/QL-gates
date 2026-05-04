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


def cart_qldit(n,d,adj_mat1,adj_mat2):
    """
    Creates the cartesian product of two QL-dits
    Parameters:
    n : number of vertices in a subgraph
    d : dimension of QL-dit
    adj_mat1 : adjacency matrix for QL-dit 1
    adj_mat2 : adjacency matrix for QL-dit 2
    """
    #adj_matrix_cart_kron = np.kron(np.eye(int((d+1)*n)),adj_mat2) + np.kron(adj_mat1,np.eye(int((d+1)*n)))
    adj_matrix_cart_kron = np.kron(np.eye(adj_mat1.shape[0]),adj_mat2) + np.kron(adj_mat1,np.eye(adj_mat2.shape[0]))
    return adj_matrix_cart_kron
