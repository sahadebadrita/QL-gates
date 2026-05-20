"""
This module contains functions for simulating quantum dynamics of spin systems using Trotter-Suzuki 
decompositions. It includes utilities for constructing Hamiltonians, generating Bell states, 
and propagating quantum states through time using specified unitary evolution operators.
The main functions are:
- `kron_power`: Computes the repeated Kronecker product of a matrix.
- `jbparams`: Generates random coupling and field parameters for a spin system and saves them as JSON files.
- `ising_hamiltonian`: Constructs the full Hamiltonian matrix for an Ising model with specified interactions and fields.
- `xymodel_two_qubit_trotter`: Builds the Trotterized unitary for a two-spin XY model.
- `transverseN`: Constructs the Trotterized unitary for a transverse Ising-like model with multiple QL-bits.
"""


import numpy as np
import logging
import json
from qlgates.gates import transform1, transform2, getRzzgate, getRyygate, getRxxgate
from qlgates.config import Config
from qlgates.helpers import kron_power
#logging.basicConfig(level=logging.INFO)

def jbparams(N, filename):
    """
    Generate random coupling and local field parameters for a spin system and save them as JSON files.

    Parameters:
    N : int - Number of sites/spins in the system.
    filename : str - Path to the directory where the parameter files will be saved.

    Returns:
    J : dict - Dictionary containing random pairwise coupling parameters for the x, y, and z channels.
    B : dict - Dictionary containing random local field parameters for the x, y, and z channels.
    """
    # --- Generate only unique (i < j) pairs ---
    pairs = [(i, j) for i in range(N) for j in range(i+1, N)]

    # --- Random couplings for each gamma channel ---
    Jx = {pair: np.random.uniform(0, 1) for pair in pairs}
    Jy = {pair: np.random.uniform(0, 1) for pair in pairs}
    Jz = {pair: np.random.uniform(0, 1) for pair in pairs}


    # --- Local field terms (site-dependent) ---
    Bx = {i: np.random.uniform(0, 1) for i in range(N)}
    By = {i: np.random.uniform(0, 1) for i in range(N)}
    Bz = {i: np.random.uniform(0, 1) for i in range(N)}

    # --- Pack into dicts (compatible with your existing function) ---
    J = {'x': Jx, 'y': Jy, 'z': Jz}
    B = {'x': Bx, 'y': By, 'z': Bz}

    with open(f'{filename}/J_params_N{N}.json', "w") as f1:
            json.dump(J, f1)
    with open(f'{filename}/B_params_N{N}.json', "w") as f2:
            json.dump(B, f2)
    return J,B

def ising_hamiltonian(J, B):
    """
    Construct the Ising Hamiltonian matrix with pairwise interactions and local field terms.

    Parameters:
    J : dict - Dictionary containing interaction matrices for the x, y, and z Pauli channels.
    B : dict - Dictionary containing local field vectors for the x, y, and z Pauli channels.

    Returns:
    H : np.ndarray - Full Hamiltonian matrix of the spin system.
    """
    # Define Pauli matrices
    I = np.eye(2, dtype=complex)
    paulis = {
        'x': np.array([[0, 1], [1, 0]], dtype=complex),
        'y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'z': np.array([[1, 0], [0, -1]], dtype=complex)
    }

    # num_qubitsumber of qubits inferred from the field vector
    N = len(B['x'])

    # Initialize Hamiltonian
    H = np.zeros((2**N, 2**N), dtype=complex)

    # Two-body interactions
    for gamma in ['x', 'y', 'z']:
        Jmat = J[gamma]
        for i in range(N):
            for j in range(i + 1, N):
                if Jmat[i, j] != 0:
                    # Construct σ_i^γ σ_j^γ
                    ops = []
                    for k in range(N):
                        if k == i or k == j:
                            ops.append(paulis[gamma])
                        else:
                            ops.append(I)
                    term = ops[0]
                    for op in ops[1:]:
                        term = np.kron(term, op)
                    H += Jmat[i, j] * term

    # Local field terms
    for gamma in ['x', 'y', 'z']:
        Bvec = B[gamma]
        for i in range(N):
            if Bvec[i] != 0:
                ops = []
                for k in range(N):
                    if k == i:
                        ops.append(paulis[gamma])
                    else:
                        ops.append(I)
                term = ops[0]
                for op in ops[1:]:
                    term = np.kron(term, op)
                H += Bvec[i] * term

    return H


def xymodel_two_qubit_trotter(n, NQL, J, dt):
    """
    Construct the second-order Trotter-Suzuki unitary for a two-spin XY model Hamiltonian.

    Parameters:
    n : int - Number of levels in each QL-dit.
    NQL : int - Number of QL-dits in the system.
    J : float - Interaction strength of the XY model.
    dt : float - Time-step used in the Trotter-Suzuki decomposition.

    Note:
    The Trotterization angles are intentionally asymmetric: `getRxxgate` is called with `J_theta / 2` and `getRyygate` is called with `J_theta`.
    This follows the standard second-order Trotter-Suzuki decomposition for the XY model, where the X interaction is split into two half steps and the Y interaction is applied as a full step.

    Returns:
    Ug : np.ndarray - Trotterized unitary matrix for the two-spin XY model evolution.
        The shape of Ug is (2*n**NQL, 2*n**NQL), where n is the number of levels per QL-dit and NQL is the number of QL-dits.
    """
    J_theta = 2 * J * dt
    rxx = getRxxgate(n, NQL, J_theta / 2)
    ryy = getRyygate(n, NQL, J_theta)
    Ug = rxx @ ryy @ rxx
    return Ug


    """
    Construct the second-order Trotter-Suzuki unitary for a two-spin transverse Ising-like Hamiltonian.

    Parameters:
    n : int - Number of levels in each QL-dit.
    NQL : int - Number of QL-dits in the system.
    J : float - Interaction strength for the ZZ interaction term.
    h : float - Transverse field strength for the onsite X term.
    dt : float - Time-step used in the Trotter-Suzuki decomposition.

    Returns:
    Ug : np.ndarray - Trotterized unitary matrix for the two-spin transverse model evolution.
    """
    #only even terms present
    Jtheta = 2*J*dt
    htheta = 2*h*(dt/2)
    Rx = transform1('Rx', n, theta=htheta, U=None)
    URx = np.kron(Rx,Rx)
    rzz = getRzzgate(n,NQL,Jtheta)
    Ug = URx @ rzz @ URx
    return Ug

def transverseN(n, NQL, J, h, dt, debug):
    """
    Construct the second-order Trotter-Suzuki unitary for a (NQL spins) transverse Ising-like model with multiple QL-bits.

    Parameters:
    n : int - Number of nodes per subgraph.
    NQL : int - Number of QL-bits in the system.
    J : float - Interaction strength for the ZZ interaction term.
    h : float - Transverse field strength for the X term.
    dt : float - Time-step used in the Trotter-Suzuki decomposition.
    debug : bool - If True, prints intermediate matrix shape information.

    Returns:
    Ug : np.ndarray
        The full Trotterized unitary matrix for the multi-QL-bit system, representing one time step of the evolution.
    """

    site_dim = 2 * n

    # --- Trotter angles ---
    Jtheta0 = 2 * J * (dt / 2.0)  # "even" bonds: interactions between QL-bits at positions (0,1), (2,3), ..., i.e., pairs where the first index is even
    Jtheta1 = 2 * J * dt          # "odd" bonds: interactions between adjacent QL-bits at odd indices, e.g., (1,2), (3,4), etc.
    htheta  = 2 * h * (dt / 2.0)  # half-step transverse field

    # --- Single QL-bit gate (Rx) and its tensor power ---
    Rx = transform1('Rx', n, theta=htheta)    # acts on one QL-bit
    URx = kron_power(Rx, NQL)                         # Rx ⊗ Rx ⊗ ... NQL times

    # --- Build two-QL-bit Rzz building blocks ---
    rzz0 = getRzzgate(n, 2, Jtheta0)  # 2-QL-bit gate
    if NQL == 2:
        # Only one pair, no "odd" layer
        Uzz0 = rzz0
        Uzz1 = np.eye(Uzz0.shape[0], dtype=complex)
    else:
        rzz1 = getRzzgate(n, 2, Jtheta1)

        if NQL % 2 == 0:
            # Even NQL: pattern like (0,1), (2,3), ...  and  (1,2), (3,4), ...
            half = NQL // 2

            # Uzz0 = rzz0 ⊗ rzz0 ⊗ ... (half times)
            Uzz0 = kron_power(rzz0, half)

            # Uzz1 = I ⊗ (rzz1 ⊗ ... ⊗ rzz1) ⊗ I
            if half > 1:
                Uzz1_core = kron_power(rzz1, half - 1)
            else:
                Uzz1_core = np.eye(site_dim**2, dtype=complex)

            Uzz1 = np.kron(np.eye(site_dim, dtype=complex),
                           np.kron(Uzz1_core, np.eye(site_dim, dtype=complex)))

        else:
            # Odd NQL: your original pattern, just expressed via kron_power
            idx = NQL // 2  # integer division

            # Uzz0: rzz0^(⊗ idx) ⊗ I
            Uzz0_core = kron_power(rzz0, idx)
            Uzz0 = np.kron(Uzz0_core, np.eye(site_dim, dtype=complex))

            # Uzz1: I ⊗ rzz1^(⊗ (idx-1))   (if idx > 1)
            if idx > 1:
                Uzz1_core = kron_power(rzz1, idx - 1)
                Uzz1 = np.kron(np.eye(site_dim, dtype=complex), Uzz1_core)
            else:
                # NQL = 3 special case: only a single rzz1 in the middle
                Uzz1 = np.kron(np.eye(site_dim, dtype=complex), rzz1)
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug(f"Shapes: {URx.shape}, {Uzz0.shape}, {Uzz1.shape}")
    # Second-order Trotter step: Rx-half, ZZ-even, ZZ-odd, ZZ-even, Rx-half
    Ug = URx @ Uzz0 @ Uzz1 @ Uzz0 @ URx
    return Ug