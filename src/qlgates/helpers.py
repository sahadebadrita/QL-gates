"""
Helper functions for quantum gate construction and manipulation.
"""
import numpy as np

def kron_power(U, reps):
    """
    Compute the repeated Kronecker product of a square matrix with itself.

    Parameters:
    U : np.ndarray - Input square matrix (shape (d, d)) to be tensor-product repeated.
    reps : int - Number of Kronecker product repetitions.

    Returns:
    Ug : np.ndarray - Matrix of shape (d**reps, d**reps) corresponding to U ⊗ U ⊗ ... ⊗ U (reps times).
    """
    if reps < 1:
        raise ValueError("reps must be >= 1")
    result = U
    for _ in range(reps - 1):
        result = np.kron(result, U)
    return result

def computational_basis_state(NQL, index):
    """
    Return the computational basis state |index> for an NQL-qubit system.
    """
    psi = np.zeros(2 ** NQL, dtype=complex)
    psi[index] = 1.0
    return psi