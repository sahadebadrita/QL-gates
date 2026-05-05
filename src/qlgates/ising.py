from qiskit import QuantumCircuit, execute,transpile, Aer, IBMQ
from qiskit.providers.aer import QasmSimulator
from qiskit.circuit.library import RZGate, RXGate, RYGate, RZZGate, RXXGate, RYYGate
from qiskit.tools.monitor import job_monitor

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import json

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

def xymodel2(J,dt):
    '''
    using the N-step second order Trotter-Suzuki decomp. to
    create the Trotterized circuit for XY model Hamiltonian
    for a system of 2 spins only

    Arg: J - interaction strength
    '''
    #only even terms present
    Jtheta = 2*J*dt
    rxx = getRxxgate(n,NQL,Jtheta)
    ryy = getRyygate(n,NQL,Jtheta)
    Ug = rxx @ ryy
    return Ug

def transverse(J,B):
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

def classical(H):
    E,V = np.linlag.eigh(H)
    return w

def expectation(psi, O):
    return np.vdot(psi, O @ psi).real

