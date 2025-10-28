
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
import networkx as nx

GATE_LIBRARY = {
    "I": lambda theta=None: np.eye(2, dtype=complex),
    "X": lambda theta=None: np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": lambda theta=None: np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": lambda theta=None: np.array([[1, 0], [0, -1]], dtype=complex),
    "H": lambda theta=None: (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex),
    "Rx": lambda theta: np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                                  [-1j*np.sin(theta/2), np.cos(theta/2)]], dtype=complex),
    "Ry": lambda theta: np.array([[np.cos(theta/2), -np.sin(theta/2)],
                                  [np.sin(theta/2), np.cos(theta/2)]], dtype=complex),
    "Rz": lambda theta: np.array([[np.exp(-1j*theta/2), 0],
                                  [0, np.exp(1j*theta/2)]], dtype=complex)
}

def getCG(gate0, gate1, n, theta=None, U=None):
	"""""
	Arguments:
	R: Adjacency matrix for the Cartesian Product of two graphs
	gate0: Conditional gate on target when condition is 0
	gate1: Conditional gate on target when condition is 1
	n: Number of nodes in each subgraph

	"""""
	#create transformation matrix for N_QL bits
	VH = get_Vg("H")	#transformation matrix
	Ucb1 = np.kron(VH, np.identity(n))
	Ucb2 = np.kron(Ucb1,Ucb1)

	Pp = np.zeros((2*n,2*n))		##create the projector of 0
	Id = np.identity(n)

	Pp[0:n,0:n] = Id
	Pp[n:2*n,0:n] = Id
	Pp[0:n,n:2*n] = Id
	Pp[n:2*n,n:2*n] = Id
	Pp *= 0.5

	Pm = Pp.copy()

	Pm[n:2*n,0:n] *= -1		##create the projector of 1
	Pm[0:n,n:2*n] *= -1

	V0 = get_Vg(gate0,theta,U)	#single qubit gate to be implemented
	V1 = get_Vg(gate1,theta,U)	#single qubit gate to be implemented

	U0 = np.kron(V0,Id)
	U1 = np.kron(V1,Id)

	UCN = np.linalg.inv(Ucb2) @ (np.kron(Pp,U0) + np.kron(Pm,U1)) @ Ucb2
	#Rg =  UCN @ R @ UCN.T.conj()
	return UCN

def getRzzgate(n,NQL,theta1):
    #Get UCN
    UCN = transform2('I', 'x', n, theta=None, U=None)
    #get Rz
    Rz = transform1('Rz', n, theta=theta1,U = None)
    URz = np.kron(np.identity((2*n)**(NQL-1)),Rz)
    URzz = UCN @ URz @ UCN
    print(np.shape(UCN),np.shape(Rz),np.shape(URz))
    return URzz

def getRyygate(n,NQL,theta1):
    URzz = getRzzgate(n,NQL,theta1)
    #Get UCN
    Rx_p = transform1('Rx', n, theta=1.57,U = None)
    Rx_m = transform1('Rx', n, theta=-1.57,U = None)
    URx_p = np.kron(Rx_p,Rx_p)
    URx_m = np.kron(Rx_m,Rx_m)
    URyy = URx_p @ URzz @ URx_m
    return URzz

def getRxxgate(n,NQL,theta1):
    URzz = getRzzgate(n,NQL,theta1)
    #get the Hadamard gate
    VH = transform1('H', n, theta=None,U = None)
    UH = np.kron(VH,VH)
    URxx = UH @ URzz @UH
    return URzz

def get_Vg(gate, theta=None, U=None):
    if gate == "U":
        # check U is unitary
        if U is None or not isinstance(U, np.ndarray) or U.shape != (2,2):
            raise ValueError("Provide a 2x2 unitary matrix U.")
        if not np.allclose(U.conj().T @ U, np.eye(2), atol=1e-10):
            raise ValueError("Matrix U is not unitary.")
        return U
    if gate not in GATE_LIBRARY:
        raise ValueError(f"Unknown gate: {gate}")
    return GATE_LIBRARY[gate](theta)

#def two_qubit(self, R, gate_type, theta=None, U=None, gate0=None, gate1=None):
def transform1_multi(gate, n, NQL, target_idx=0, theta=None, U=None):
    """
    Apply a single QL-bit gate transformation to one QL-bit
    in a system of NQL QL-bits.

    Parameters
    ----------
    gate : str
        Type of gate ("x", "y", "z", "H", "Rx", "Rz", etc.)
    n : int
        Number of nodes per subgraph
    NQL : int
        Total number of QL-bits in the system
    target_idx : int
        Index of the QL-bit to which the gate is applied (0-indexed)
    theta : float, optional
        Rotation angle (if needed for gate)
    U : np.ndarray, optional
        Custom 2×2 unitary for arbitrary gates

    Returns
    -------
    np.ndarray
        The full transformation matrix acting on all QL-bits
    """

    # Base single-QL-bit operators
    VH = get_Vg("H")
    Vg = get_Vg(gate, theta, U)
    Iq = np.eye(2)       # identity in QL-bit space
    In = np.eye(n)       # identity in subgraph space

    # (V_H ⊗ I_n)
    VH_block = np.kron(VH, In)
    # (V_g ⊗ I_n)
    Vg_block = np.kron(Vg, In)

    # Build tensor products across NQL bits
    Ucb_list = []
    Ug_list = []

    for i in range(NQL):
        if i == target_idx:
            Ug_list.append(Vg_block)
        else:
            Ug_list.append(np.kron(Iq, In))
        Ucb_list.append(VH_block)

    # Full Kronecker products
    Ucb_total = Ucb_list[0]
    Ug_total = Ug_list[0]
    for i in range(1, NQL):
        Ucb_total = np.kron(Ucb_total, Ucb_list[i])
        Ug_total = np.kron(Ug_total, Ug_list[i])

    # Apply transformation
    U_total = np.linalg.inv(Ucb_total) @ Ug_total @ Ucb_total

    return U_total

def cartesian_product_igraph(adj_list, directed=False, return_matrix=False):
    """
    Compute the Cartesian product of multiple adjacency matrices using igraph.

    Parameters
    ----------
    adj_list : list of np.ndarray
        List of adjacency matrices [A1, A2, ..., Am]
    directed : bool, optional
        Whether to treat graphs as directed (default: False)
    return_matrix : bool, optional
        If True, also return the final adjacency matrix

    Returns
    -------
    ig.Graph
        The Cartesian product graph G_total
    np.ndarray (optional)
        The adjacency matrix of G_total (if return_matrix=True)
    """

    if len(adj_list) < 2:
        raise ValueError("Need at least two adjacency matrices for Cartesian product.")

    # --- Step 1: Convert adjacency matrices to igraph.Graph objects ---
    graphs = []
    for A in adj_list:
        g = nx.from_numpy_array(A)
        #g = ig.Graph.Adjacency((A > 0).tolist(), mode=ig.ADJ_DIRECTED if directed else ig.ADJ_UNDIRECTED)
        graphs.append(g)

    # --- Step 2: Take Cartesian products iteratively ---
    G_total = graphs[0]
    for G in graphs[1:]:
        G_total = nx.cartesian_product(G_total,G)

    # --- Step 3: Optionally return adjacency matrix ---
    if return_matrix:
        A_total = nx.to_numpy_array(G_total)
        print(A_total.shape)
        return G_total, A_total
    else:
        return G_total

