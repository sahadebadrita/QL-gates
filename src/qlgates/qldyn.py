import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import json
from qlgates.gates import transform1, getRzzgate

def jbparams(N,filename):
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

def ising_hamiltonian(J,B):
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


def xymodel2(n,NQL,J,dt):
    '''
    using the N-step second order Trotter-Suzuki decomp. to
    create the Trotterized circuit for XY model Hamiltonian
    for a system of 2 spins only

    Arg: J - interaction strength
    '''
    #only even terms present
    Jtheta = 2*J*dt
    rxx = getRxxgate(n,NQL,Jtheta/2)
    ryy = getRyygate(n,NQL,Jtheta)
    Ug = rxx @ ryy @ rxx
    return Ug

def transverse2(n,NQL,J,h,dt):
    '''
    using the N-step second order Trotter-Suzuki decomp. to
    create the Trotterized circuit for transverse model Hamiltonian
    for a system of 2 spins only
    Arg: J - interaction strength
    h - onsite interaction term
    '''
    #only even terms present
    Jtheta = 2*J*dt
    htheta = 2*h*(dt/2)
    Rx = transform1('Rx', n, theta=htheta,U = None)
    URx = np.kron(Rx,Rx)
    rzz = getRzzgate(n,NQL,Jtheta)
    Ug = URx @ rzz @ URx
    return Ug

def kron_power(U, reps):
    """
    Compute U ⊗ U ⊗ ... ⊗ U (reps times).
    If reps = 1, returns U.
    """
    if reps < 1:
        raise ValueError("reps must be >= 1")
    Ug = U
    for _ in range(reps - 1):
        Ug = np.kron(Ug, U)
    return Ug


def transverseN(n, NQL, J, h, dt,debug):
    """
    Second-order Trotter-Suzuki step for transverse Ising-like model
    in the QL setting, generalized to NQL QL-bits.

    Args
    ----
    n    : number of nodes per subgraph (local dim = 2n)
    NQL  : number of QL-bits ("spins")
    J    : interaction strength (ZZ term)
    h    : transverse field strength (X term)
    dt   : Trotter time step

    Returns
    -------
    Ug : np.ndarray
        Full Trotter step unitary on (2n)^NQL-dimensional space
    """

    d_site = 2 * n

    # --- Trotter angles ---
    Jtheta0 = 2 * J * (dt / 2.0)  # "even" bonds
    Jtheta1 = 2 * J * dt          # "odd" bonds
    htheta  = 2 * h * (dt / 2.0)  # half-step transverse field

    # --- Single QL-bit gate (Rx) and its tensor power ---
    Rx = transform1('Rx', n, theta=htheta, U=None)    # acts on one QL-bit
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
                Uzz1_core = np.eye(d_site**2, dtype=complex)

            Uzz1 = np.kron(np.eye(d_site, dtype=complex),
                           np.kron(Uzz1_core, np.eye(d_site, dtype=complex)))

        else:
            # Odd NQL: your original pattern, just expressed via kron_power
            idx = NQL // 2  # integer division

            # Uzz0: rzz0^(⊗ idx) ⊗ I
            Uzz0_core = kron_power(rzz0, idx)
            Uzz0 = np.kron(Uzz0_core, np.eye(d_site, dtype=complex))

            # Uzz1: I ⊗ rzz1^(⊗ (idx-1))   (if idx > 1)
            if idx > 1:
                Uzz1_core = kron_power(rzz1, idx - 1)
                Uzz1 = np.kron(np.eye(d_site, dtype=complex), Uzz1_core)
            else:
                # NQL = 3 special case: only a single rzz1 in the middle
                Uzz1 = np.kron(np.eye(d_site, dtype=complex), rzz1)
    if debug:
        # Debug (optional)
        print("Shapes:", URx.shape, Uzz0.shape, Uzz1.shape)

    Ug = URx @ Uzz0 @ Uzz1 @ Uzz0 @ URx
    return Ug

def propagate_state(cfg,psi):
    print('Propagate_state')
    Ntot = (2 * cfg.n) ** cfg.NQL
    Ug = transverseN(cfg.n, cfg.NQL, cfg.J, cfg.h, cfg.deltat,debug=False)
    #Ug = xymodel2(cfg.n,cfg.NQL,cfg.J,cfg.deltat)
    psit = np.empty((Ntot,cfg.timesteps),dtype=complex)
    psit[:,0] = psi
    print(cfg.timesteps)
    for step in range(1,cfg.timesteps,1):
        print('yyyy')
        print("time", step*cfg.deltat,flush=True)
        psit[:,step] = Ug @ psit[:,step-1]

    return psit

def bell_state1(cfg,psi,idx):
    Ntot = (2 * cfg.n) ** cfg.NQL
    UH = transform1('H', cfg.n, theta=None, U=None)
    UI = transform1('I', cfg.n, theta=None, U=None)
    UCN = transform2("I", "x", cfg.n, theta=None, U=None)
    if idx == 'I':
        print('I')
        #psit = UCN @ np.kron(UH,UI) @ psi
        psit = psi
    elif idx == 'x':
        print('x')
        Ux = transform1('x', cfg.n, theta=None, U=None)
        #psit = UCN @ np.kron(UH,UI) @ np.kron(Ux,UI) @ psi
        psit = np.kron(Ux,UI) @ psi
    elif idx == 'z':
        print('z')
        Uz = transform1('z', cfg.n, theta=None, U=None)
        #psit = UCN @ np.kron(UH,UI) @ np.kron(Uz,UI) @ psi
        psit = np.kron(Uz,UI) @ psi
    elif idx == 'y':
        print('y')
        Ux = transform1('x', cfg.n, theta=None, U=None)
        Uz = transform1('z', cfg.n, theta=None, U=None)
        #psit = UCN @ np.kron(UH,UI) @ np.kron((Uz @ Ux),UI) @ psi
        psit = np.kron((Uz @ Ux),UI) @ psi

    return psit
