"""
Helper functions for quantum gate construction and manipulation.
"""
import numpy as np
from qlgates.gates import transform1

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

def kron_list(mats):
    """Kronecker product of a list of sparse matrices."""
    K = mats[0]
    for M in mats[1:]:
        K = np.kron(K, M)
    return K

def build_local_operators(UO, UI, N):
    """
    Construct Uz0, Uz1, ..., Uz(N-1) for N subsystems.

    UO : operator acting on one site
    UI : identity operator on one site
    N  : number of sites (NQL)

    Returns:
        list of sparse matrices [Uz0, Uz1, ..., Uz(N-1)]
    """

    ops = []

    for k in range(N):
        mats = [UI] * N
        mats[k] = UO
        ops.append(kron_list(mats))

    return ops

def build_observable(cfg, gate_obs, site):
    """
    Build the local Z observable for a given site in an NQL-qubit system.
    """
    UO = transform1(gate_obs, cfg.n, theta=None, U=None)
    UI = transform1('I', cfg.n, theta=None, U=None)
    UO_ops = build_local_operators(UO, UI, cfg.NQL)
    return UO_ops[site]

def expectation(psi, O):
    """
    Calculate the expectation value of operator O in state psi.
    Parameters:
    psi : np.ndarray - State vector(s) of shape (d, m) where d is the Hilbert space dimension and m is the number of states.
    O : np.ndarray - Operator matrix of shape (d, d).
    Returns:
    expectation : np.ndarray - Array of shape (m,) containing the expectation value for each state.
    """    
    m = psi.shape[1]
    expectation = np.empty(m)
    for i in range(m):
        expectation[i] = np.vdot(psi[:, i], O @ psi[:, i]).real
    return expectation

def loschmidt_amplitude(cfg,psi0, psit):
    """
    Taken from the paper "Dynmaical Quantum Phase Transitions: a review" by Markus Heyl, arXiv:1801.07016.
    Calculate the Loschmidt amplitude <psi0|psit>.
    Parameters:
    psi0 : np.ndarray - Initial state vector of shape (d,).
    psit : np.ndarray - Final state vector of shape (d,timesteps).
    Returns:
    amplitude : complex - The Loschmidt amplitude.
    """
    G = np.zeros(psit.shape[1], dtype=complex)
    for i in range(psit.shape[1]):
        G[i] = np.vdot(psi0, psit[:, i])
    L = np.abs(G) ** 2
    rate = -(1/cfg.NQL)*np.log(L)
    return L, rate