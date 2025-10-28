#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for Quantum-Like (QL) graph gate simulations
Author: Deba and Ethan
Date: [YYYY-MM-DD]

This script builds QL adjacency matrices, applies two-qubit gate transformations,
and sets up parameters for dynamical simulations of QL-bit systems.
"""

# ============================================================
# Imports
# ============================================================
import os
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import expm, logm
from scipy.integrate import solve_ivp
from scipy.stats import unitary_group
import igraph as ig
import networkx as nx

# Local module imports
from utils import qldit, cart_qldit, transform1, transform2 # ensure utils.py exists in the same directory
from gateslib import transform1_multi, cartesian_product_igraph
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
n = 4          # number of nodes in subgraph
k = 3           # degree
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
NQL = 4
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
    #PRINT statements:
    print(datetime.datetime.now())
    print("\n--- QL Resource Parameters ---")
    print(f"n (nodes per subgraph)   : {n}")
    print(f"k (degree)               : {k}")
    print(f"d                        : {d}")
    print(f"l (coupling length)      : {l}")
    print(f"lp (alternate coupling)  : {lp}")
    print(f"nsamp (samples)          : {nsamp}")
    print(f"Cartesian Product        : {CartPdt}")
    print(f"Coupling enabled         : {coupling}")
    print(f"Periodic boundary cond.  : {periodic}")
    print(f"Full connectivity        : {full}")
    print("------------------------------\n")
    print("Building QL resource")
    R = qldit(n=n, k=k, d=d, l=l, coupling=coupling,
              periodic=periodic, full=full)
    print("Applying single-qubit gate transformation on single QLbit")
    #UCN = transform2('I', 'x', n, theta=None, U=None)
    Ug = transform1('H',n)
    Rg = Ug @ R @ Ug.T.conj()
    # Optionally visualize or inspect the transformed adjacency
    if CHECK:
        print("Shape of transformed adjacency:", Rg.shape)
        print("Hermitian check:", np.allclose(Rg, Rg.T.conj(), atol=1e-10))
        print("Unitary check:", np.allclose((Ug @  Ug.T.conj()), np.eye(Ug.shape[0], dtype=Ug.dtype), atol=1e-10))

    if CartPdt and NQL == 2:
        R2 = qldit(n=n, k=k, d=d, l=lp, coupling=coupling,
                   periodic=periodic, full=full)
        Rg2 = cart_qldit(n=n, d=d, adj_mat1=Rg, adj_mat2=R2)
        R12 = cart_qldit(n=n, d=d, adj_mat1=R, adj_mat2=R2)

        print("Gate transformation on CP of 2 QLbits: R1g x I + I x R2")
        # Optionally visualize or inspect the transformed adjacency
        if CHECK:
            print("Shape of transformed adjacency:", Rg2.shape)
            print("Hermitian check:", np.allclose(Rg2, Rg2.T.conj(), atol=1e-10))

        print("Applying single-qubit gate transformation on CP of 2 QLbits")
        Ug12 = transform1_multi('H', n, NQL, target_idx=0, theta=None, U=None)
        Rg12 = Ug12 @ R12 @ Ug12.T.conj()
        # Optionally visualize or inspect the transformed adjacency
        if CHECK:
            print("Shape of transformed adjacency:", Rg12.shape)
            print("Shape of transformation matrix:", Ug12.shape)
            print("Hermitian check:", np.allclose(Rg12, Rg12.T.conj(), atol=1e-10))
            print("Unitary check:", np.allclose((Ug12 @  Ug12.T.conj()), np.eye(Ug12.shape[0], dtype=Ug12.dtype), atol=1e-10))
        print('Diff b/w Rg2 and Rg12',np.allclose(Rg12,Rg2,atol=1e-10))
        print('Diff b/w Rg2 and Rg12',np.linalg.norm(Rg12-Rg2,'fro'))

    # Compute Cartesian product using igraph
    G_total, A_total = cartesian_product_igraph([R, R, R, R], return_matrix=True)
    #print('Shape of adjacency matrix:',A_total.shape())

    Ug123 = transform1_multi('H', n, NQL, target_idx=0, theta=None, U=None)
    Rg123 = Ug123 @ A_total @ Ug123.T.conj()
    if CHECK:
        print("Shape of transformed adjacency:", Rg123.shape)
        print("Shape of transformation matrix:", Ug123.shape)
        print("Hermitian check:", np.allclose(Rg123, Rg123.T.conj(), atol=1e-10))
        print("Unitary check:", np.allclose((Ug123 @  Ug123.T.conj()), np.eye(Ug123.shape[0], dtype=Ug123.dtype), atol=1e-10))
    print(datetime.datetime.now())
