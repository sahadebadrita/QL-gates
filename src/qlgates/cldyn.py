import numpy as np
from scipy.linalg import expm

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
        H += -J * kron_N(ops)

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

def mz_plot(mz_site):
    fig, axes = plt.subplots(2, 2, figsize=(8,6), sharex=True)
    
    axes = axes.ravel()
    times = np.arange(mz_site.shape[0]) #check this
    axes[0].plot(times, mz_site[:,0], color="#2a4d69")
    axes[0].set_title(r"$\langle \sigma^z_0 \rangle$")
    axes[0].set_ylabel("Magnetization")
    
    axes[1].plot(times, mz_site[:,1], color="#d7263d")
    axes[1].set_title(r"$\langle \sigma^z_1 \rangle$")
    
    axes[2].plot(times, mz_site[:,2], color="#1f4d34")
    axes[2].set_title(r"$\langle \sigma^z_2 \rangle$")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Magnetization")
    
    axes[3].plot(times, mz_site[:,3], color="k", linestyle="--", linewidth=2)
    axes[3].set_title("Average M")
    axes[3].set_xlabel("Time")
    
    fig.suptitle("Local magnetization")
    
    plt.tight_layout()
    plt.savefig(f'mz_h{h}_J{J}_classical_N{N}.png')
    #plt.savefig(f'newtrotter_h{h}_dt{dt}_steps{n_steps}.png')
