import logging

import numpy as np
from scipy.linalg import expm
from qlgates.helpers import kron_power
from qlgates.config import Config
from qlgates.gates import get_Vg, rzz_matrix

def transverse_ising_trotter(NQL, J, h, deltat):
    """
    Classical simulation of second-order Trotter-Suzuki unitary for
    the transverse-field Ising Hamiltonian:
        H = -J * sum_i Z_i Z_{i+1}  +  h * sum_i X_i

    Trotter structure (identical to your transverseN):
        Rx(h·dt/2) · Rzz_even(J·dt) · Rzz_odd(J·dt) · Rzz_even(J·dt) · Rx(h·dt/2)

    Parameters
    ----------
    NQL    : int   - Number of qubits.
    J      : float - ZZ interaction strength.
    h      : float - Transverse field strength.
    deltat : float - Trotter time step.

    Returns
    -------
    Ug : np.ndarray  shape (2**NQL, 2**NQL)
        Full Trotterized unitary for one time step.
    """

    site_dim = 2  # qubit

    # --- Trotter angles (same convention as your code) ---
    Jtheta0 = 2 * J * (deltat / 2.0)   # even-bond half-step
    Jtheta1 = 2 * J * deltat            # odd-bond full step
    htheta  = 2 * h * (deltat / 2.0)   # transverse field half-step

    # --- Single-qubit Rx and its tensor power ---
    Rx  = get_Vg('Rx',htheta,U=None)
    URx = kron_power(Rx, NQL)           # Rx ⊗ Rx ⊗ ... NQL times

    # --- Two-qubit Rzz building blocks ---
    rzz0 = rzz_matrix(Jtheta0)          # even-bond gate
    if NQL == 2:
        Uzz0 = rzz0
        Uzz1 = np.eye(Uzz0.shape[0], dtype=complex)
    else:
        rzz1 = rzz_matrix(Jtheta1)      # odd-bond gate

        if NQL % 2 == 0:
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
            # Odd NQL
            idx = NQL // 2

            # Uzz0: rzz0^(⊗ idx) ⊗ I
            Uzz0_core = kron_power(rzz0, idx)
            Uzz0 = np.kron(Uzz0_core, np.eye(site_dim, dtype=complex))

            # Uzz1: I ⊗ rzz1^(⊗ (idx-1))
            if idx > 1:
                Uzz1_core = kron_power(rzz1, idx - 1)
                Uzz1 = np.kron(np.eye(site_dim, dtype=complex), Uzz1_core)
            else:
                # NQL = 3: single rzz1 in the middle
                Uzz1 = np.kron(np.eye(site_dim, dtype=complex), rzz1)

    # Second-order Trotter: Rx-half · Rzz-even · Rzz-odd · Rzz-even · Rx-half
    Ug = URx @ Uzz0 @ Uzz1 @ Uzz0 @ URx
    return Ug

def exact_unitary(NQL, J, h, deltat):
    """
    Exact U = exp(-i H dt) via direct matrix exponentiation, for comparison.
    H = -J * sum_i Z_i Z_{i+1}  +  h * sum_i X_i   (open chain)
    """
    dim = 2 ** NQL
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    H = np.zeros((dim, dim), dtype=complex)

    # ZZ terms
    for i in range(NQL - 1):
        ops = [I] * NQL
        ops[i]     = Z
        ops[i + 1] = Z
        ZZi = ops[0]
        for op in ops[1:]:
            ZZi = np.kron(ZZi, op)
        H += J * ZZi

    # X terms
    for i in range(NQL):
        ops = [I] * NQL
        ops[i] = X
        Xi = ops[0]
        for op in ops[1:]:
            Xi = np.kron(Xi, op)
        H += h * Xi
    U_exact = expm(-1j * H * deltat)    
    return U_exact

def propagate_state_classical(cfg:Config, psi: np.ndarray) -> np.ndarray:
    #!!!Add unitary argument (Ug) to cfg and pass it here instead of building it inside this function. This way we can test with different unitaries (e.g. identity for norm preservation test).
    """
    Propagate an initial quantum state through time using the hamiltonian model evolution operator.

    Parameters:
    cfg : object - Configuration object containing system and simulation parameters.
    psi : np.ndarray - Initial state vector of the system.

    Returns:
    psit : np.ndarray - Time-evolved state vectors for all simulation time steps.
    """
    logging.info('Propagate_state')
    Ntot = 2 ** cfg.NQL
    Ug = transverse_ising_trotter(cfg.NQL, cfg.J, cfg.h, cfg.deltat)
    #Ug = build_unitary(cfg)
    psit = np.empty((Ntot,cfg.timesteps),dtype=complex)
    psit[:,0] = psi
    logging.info(f"Propagating {cfg.timesteps} time steps", flush=True)
    for step in range(1,cfg.timesteps,1):
        logging.info(f"Time step {step}/{cfg.timesteps}", flush=True)
        psit[:,step] = Ug @ psit[:,step-1]

    return psit

def bell_state_qubit(cfg: Config, psi0, kind="phi_plus"):

    """
    Generate a Bell basis state from the initial |00⟩ state.

    Parameters:
    cfg : object - Configuration object containing system parameters.
    psi0 : np.ndarray - Initial state vector, expected to represent the |00...0⟩ state.
    kind : str - Type of Bell state to generate ("phi_plus", "phi_minus", "psi_plus", or "psi_minus").
        'phi_plus'  -> |Φ⁺⟩
        'phi_minus' -> |Φ⁻⟩
        'psi_plus'  -> |Ψ⁺⟩
        'psi_minus' -> |Ψ⁻⟩

    Returns:
    state : np.ndarray - Generated Bell state vector.
    """

    H = get_Vg("H", theta=None, U=None)
    I = get_Vg("I", theta=None, U=None)
    CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])

    # Bell entangling circuit
    Ubell = CNOT @ np.kron(H, I)
    
    # Generate other Bell states via local ops on qubit 0
    ops = {
        "phi_plus": np.kron(I, I),
        "phi_minus": np.kron(I,get_Vg("z", theta=None, U=None)),
        "psi_plus": np.kron(get_Vg("x", theta=None, U=None), I),
        "psi_minus": np.kron(get_Vg("x", theta=None, U=None) , get_Vg("z", theta=None, U=None))
    }

    if kind not in ops:
        raise ValueError(f"Unknown Bell state: {kind}")
    Ug = ops[kind] @ Ubell
    psi_bell = Ug @ psi0
    return psi_bell

def bell_state_qubitrl(cfg: Config, psi0, kind="phi_plus"):

    """
    Generate a Bell basis state from the initial |00⟩ state.

    Parameters:
    cfg : object - Configuration object containing system parameters.
    psi0 : np.ndarray - Initial state vector, expected to represent the |00...0⟩ state.
    kind : str - Type of Bell state to generate ("phi_plus", "phi_minus", "psi_plus", or "psi_minus").
        'phi_plus'  -> |Φ⁺⟩
        'phi_minus' -> |Φ⁻⟩
        'psi_plus'  -> |Ψ⁺⟩
        'psi_minus' -> |Ψ⁻⟩

    Returns:
    state : np.ndarray - Generated Bell state vector.
    """

    H = get_Vg("H", theta=None, U=None)
    I = get_Vg("I", theta=None, U=None)
    CNOT = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 1, 0, 0]
])

    # Bell entangling circuit
    Ubell = CNOT @ np.kron(I, H)
    
    # Generate other Bell states via local ops on qubit 0
    ops = {
        "phi_plus": np.kron(I, I),
        "phi_minus": np.kron(get_Vg("z", theta=None, U=None),I),
        "psi_plus": np.kron(I,get_Vg("x", theta=None, U=None)),
        "psi_minus": np.kron(get_Vg("z", theta=None, U=None), get_Vg("x", theta=None, U=None))
    }

    if kind not in ops:
        raise ValueError(f"Unknown Bell state: {kind}")
    Ug = ops[kind] @ Ubell
    psi_bell = Ug @ psi0
    return psi_bell    

sx = np.array([[0, 1],
               [1, 0]], dtype=complex)

sz = np.array([[1,  0],
               [0, -1]], dtype=complex)

id2 = np.eye(2, dtype=complex)

def kron_N(ops):
    """
    Kronecker product of a list of operators.

    Parameters:
    ops (list): List of operators (numpy arrays) to take the Kronecker product of.
    Returns:
    numpy.ndarray: The Kronecker product of the input operators.
    """
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out

def transverse_field_ising(N, J, h):
    """
    Build the transverse field Ising model (TFIM) Hamiltonian for N spins.
    Open boundary conditions are assumed.

    Parameters:
    N (int): Number of spins.
    J (float): Coupling strength.
    h (float): Transverse field strength.

    Returns:
    numpy.ndarray: The TFIM Hamiltonian.
    """
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)

    # --- Ising ZZ term ---
    for i in range(N - 1):
        ops = [id2] * N
        ops[i]     = sz
        ops[i + 1] = sz
        H += -J * kron_N(ops)

    # --- Transverse field X term ---
    for i in range(N):
        ops = [id2] * N
        ops[i] = sx
        H += h * kron_N(ops)

    return H

def initial_state_z_up(N):
    """
    Create the initial state |00...0> (all spins up in z-basis).

    Parameters:
    N (int): Number of spins.
    Returns:
    numpy.ndarray: The initial state vector.
    """

    psi0 = np.array([1, 0], dtype=complex)
    for _ in range(N - 1):
        psi0 = np.kron(psi0, np.array([1, 0], dtype=complex))
    return psi0

def time_evolve(H, psi0, t):
    """
    Evolve the state psi0 under Hamiltonian H for time t.

    Parameters:
    H (numpy.ndarray): Hamiltonian matrix.
    psi0 (numpy.ndarray): Initial state vector.
    t (float): Time.
    Returns:
    numpy.ndarray: Evolved state vector.
    """
    U = expm(-1j * H * t)
    return U @ psi0

def evolve_times(H, psi0, times):
    """
    Evolve the state psi0 under Hamiltonian H for a list of times.

    Parameters:
    H (numpy.ndarray): Hamiltonian matrix.
    psi0 (numpy.ndarray): Initial state vector.
    times (list): List of times.
    Returns:
    list: List of evolved state vectors.
    """
    states = []
    for t in times:
        states.append(expm(-1j * H * t) @ psi0)
    return states

def local_sz(N, site):
    """
    Construct the local Sz operator for a given site in an N-spin system.

    Parameters:
    N (int): Number of spins.
    site (int): Site index.
    Returns:
    numpy.ndarray: The local Sz operator for the specified site.
    """
    ops = [id2] * N
    ops[site] = sz
    return kron_N(ops)

def expectation(psi, O):
    """
    Calculate the expectation value of operator O in state psi.

    Parameters:
    psi (numpy.ndarray): State vector.
    O (numpy.ndarray): Operator matrix.
    Returns:
    float: The expectation value.
    """
    return np.vdot(psi, O @ psi).real

def H_ZZ(N, J):
    """
    Construct the Ising ZZ interaction term for N spins.
    Open boundary conditions are assumed.
    Parameters:
    N (int): Number of spins.
    J (float): Coupling strength.
    Returns:
    numpy.ndarray: The Ising ZZ interaction Hamiltonian.    
    """

    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(N - 1):
        ops = [id2] * N
        ops[i]     = sz
        ops[i + 1] = sz
        H += J * kron_N(ops)

    return H

def H_X(N, h):
    """
    Construct the transverse field X term for N spins.
    
    Parameters:
    N (int): Number of spins.
    h (float): Transverse field strength.
    Returns:
    H (numpy.ndarray): The transverse field X Hamiltonian.
    """
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)

    for i in range(N):
        ops = [id2] * N
        ops[i] = sx
        H += -h * kron_N(ops)

    return H

def trotter_step(psi, U_ZZ_half, U_X):
    """
    Perform a single Trotter step: U_ZZ_half -> U_X -> U_ZZ_half.
    Parameters:
    psi (numpy.ndarray): Current state vector.
    U_ZZ_half (numpy.ndarray): Half-step Ising ZZ evolution operator.
    U_X (numpy.ndarray): Transverse field evolution operator.
    Returns:
    numpy.ndarray: Evolved state vector.
    """
    psi = U_ZZ_half @ psi
    psi = U_X @ psi
    psi = U_ZZ_half @ psi
    return psi

def trotter_evolve(N, J, h, psi0, dt, n_steps):
    """
    Second-order Trotter-Suzuki evolution.
    Parameters:
    N (int): Number of spins.
    J (float): Coupling strength.
    h (float): Transverse field strength.
    psi0 (numpy.ndarray): Initial state vector.
    dt (float): Time step size.
    n_steps (int): Number of Trotter steps.
    Returns: 
    states (list): List of state vectors at each time step.
    """
    Hzz = H_ZZ(N, J)
    Hx  = H_X(N, h)

    U_ZZ_half = expm(-1j * Hzz * dt / 2)
    U_X       = expm(-1j * Hx  * dt)

    psi = psi0.copy()
    states = [psi]

    for _ in range(n_steps):
        psi = trotter_step(psi, U_ZZ_half, U_X)
        states.append(psi)

    return states