#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for Quantum-Like (QL) graph gate simulations
Author: [Your Name]
Date: [YYYY-MM-DD]

This script builds QL adjacency matrices, applies two-qubit gate transformations,
and sets up parameters for dynamical simulations of QL-bit systems.
"""

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

# Local module imports
from utils import qldit, cart_qldit, transform2 # ensure utils.py exists in the same directory

# ============================================================
# Flags & Configuration
# ============================================================
CHECK = True   # Enable sanity checks or debugging

# ============================================================
# Simulation Parameters
# ============================================================

# --- Time grid for QL oscillations ---
dt_q = 10
Tmin_q = 0
Tmax_q = 500
Nt_q = int((Tmax_q - Tmin_q) / dt_q) + 1
t_q = np.linspace(Tmin_q, Tmax_q, Nt_q)  # in a.u. (1 a.u. = 0.0242 fs)

# --- Ising model parameters ---
B = [1.0]
J = [0.8]

# --- QL resource parameters ---
n = 12          # number of nodes in subgraph
k = 8           # degree
d = 1
l = 1
lp = 2
nsamp = 100
CartPdt = True
coupling = True
periodic = False
full = False

# --- Dynamics parameters ---
Gamma = 10
sigma = 0
dt = 0.01
Tmin = 0
Tmax = 10 + dt
NQL = 2
Ntot = (2 * n) ** NQL
Nt = int((Tmax - Tmin) / dt)
t_eval = np.linspace(Tmin, Tmax, Nt)
omega_const = 0.5
omega = np.full(Ntot, omega_const)

thr_spec = 0.1  # tolerance for spectral decomposition check

# --- Arrays for dynamics ---
trtrt = np.zeros((Ntot, Nt, Nt_q), dtype=complex)
x_diag = np.zeros((Ntot, Nt, Nt_q), dtype=complex)

# --- Plotting parameters ---
colormap_phi = cm.twilight
nbins = 50

# ============================================================
# Helper: Apply two-qubit transformation
# ============================================================
def apply_two_qubit_gate(R):
    """
    Apply a controlled gate transformation (example: Controlled-X)
    to the QL resource adjacency matrix.
    """
    UCN = transform2('I', 'x', n, theta=None, U=None)
    Rg_CN = UCN @ R @ UCN.T.conj()
    return Rg_CN

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    print("Building QL resource")
    R = qldit(n=n, k=k, d=d, l=l, coupling=coupling,
              periodic=periodic, full=full)
    if CartPdt:
        R2 = qldit(n=n, k=k, d=d, l=lp, coupling=coupling,
                   periodic=periodic, full=full)
        R = cart_qldit(n=n, d=d, adj_mat1=R, adj_mat2=R2)

    print("Applying two-qubit gate transformation...")
    UCN = transform2('I', 'x', n, theta=None, U=None)
    Rg_CN = UCN @ R @ UCN.T.conj()

    # Optionally visualize or inspect the transformed adjacency
    if CHECK:
        print("Shape of transformed adjacency:", Rg_CN.shape)
        print("Hermitian check:", np.allclose(Rg_CN, Rg_CN.T.conj(), atol=1e-10))
        print("Unitary check:", np.allclose((UCN @  UCN.T.conj()), np.eye(UCN.shape[0], dtype=UCN.dtype), atol=1e-10))


