import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.linalg import logm as logm
from scipy.linalg import expm as expm
from scipy.integrate import solve_ivp
from scipy.stats import unitary_group

#Scripts that relate to gate transformations -- both single and two qubit

def get_Vg(gate, theta=None, U=None):
    """
    Gets the single qubit gates
    Parameters:
    gate: string with gate symbol
    theta: arg for rotation gate
    U: exact unitary for check
    """
    Vg = np.zeros((2,2), dtype=complex)

    if gate == "U":
        if U is None:
            raise ValueError("You must provide a 2x2 unitary matrix U for 'Ucustom'.")
        if not isinstance(U, np.ndarray) or U.shape != (2, 2):
            raise ValueError("U must be a 2x2 NumPy array.")
        if not np.allclose(U.conj().T @ U, np.eye(2), atol=1e-10):
            raise ValueError("Provided matrix U is not unitary.")
        return U
    if gate == "I":
        Vg[0,0] = 1.
        Vg[0,1] = 0.
        Vg[1,0] = 0.
        Vg[1,1] = 1.

    elif gate == "H":
        Vg[0,0] = 1.
        Vg[0,1] = 1
        Vg[1,0] = 1
        Vg[1,1] = -1

        Vg /= np.sqrt(2.)

    elif gate == "x":
        Vg[0,0] = 0.
        Vg[0,1] = 1.
        Vg[1,0] = 1.
        Vg[1,1] = 0.

    elif gate == "y":
        Vg[0,0] = 0.
        Vg[0,1] = -1j
        Vg[1,0] = 1j
        Vg[1,1] = 0

    elif gate == "z":
        Vg[0,0] = 1.
        Vg[0,1] = 0
        Vg[1,0] = 0
        Vg[1,1] = -1

    elif gate == "Rz":
        if theta is None:
            raise ValueError("Theta must be provided for Rz gate.")
        Vg[0,0] = np.exp(-1j * theta / 2)
        Vg[1,1] = np.exp(1j * theta / 2)
        Vg[0,1] = 0.
        Vg[1,0] = 0.

    elif gate == "Ry":
        if theta is None:
            raise ValueError("Theta must be provided for Ry gate.")
        Vg[0,0] = np.cos(theta/2)
        Vg[0,1] = -np.sin(theta/2)
        Vg[1,0] = np.sin(theta/2)
        Vg[1,1] = np.cos(theta/2)

    elif gate == "Rx":
        if theta is None:
            raise ValueError("Theta must be provided for Rx gate.")
        Vg[0,0] = np.cos(theta/2)
        Vg[0,1] = -1j * np.sin(theta/2)
        Vg[1,0] = -1j * np.sin(theta/2)
        Vg[1,1] = np.cos(theta/2)

    else:
        print("wrong gate!")
        # sys.exit()

    return Vg


#single qubit gate implementation
def transform_R(R, gate, n, theta=None, U=None):

	VH = get_Vg("H")
	Vg = get_Vg(gate,theta,U)
	Ucb = np.kron(VH, np.identity(n))
	Ug =  (np.linalg.inv(Ucb) @
	np.kron(Vg,np.identity(n)) @ Ucb)

	Rg =  Ug @ R @ Ug.T.conj()

	return Rg

# two qubit gate implementation
def transform_R12(R, gate0, gate1, n, theta=None, U=None):
	"""""
	Arguments:
	R: Adjacency matrix for the Cartesian Product of two graphs
	gate0: Conditional gate on target when condiion is 0
	gate1: Conditional gate on target when condiion is 1
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
	Rg =  UCN @ R @ UCN.T.conj()
	return Rg

def transform1(gate, n, theta=None, U=None):
    """
    Single QL-bit gate matrix.

    Parameters
    ----------
    gate : str
        Gate name.
    n : int
        Number of nodes in QL-bit subgraph.
    theta : float, optional
        Argument for arbitrary-angle gates.
    U : np.ndarray, optional
        Optional unitary.
    """

    VH = get_Vg("H")
    Vg = get_Vg(gate, theta, U)

    Ucb = np.kron(VH, np.identity(n))

    Ug = (
        np.linalg.inv(Ucb)
        @ np.kron(Vg, np.identity(n))
        @ Ucb
    )

    return Ug

def transform2(gate0, gate1, n, theta=None, U=None):
    """
    Arguments:
        gate0 : Conditional gate on target when condition is 0
        gate1 : Conditional gate on target when condition is 1
        n     : Number of nodes in each subgraph
        theta : Optional parameter
        U     : Optional unitary
    """

    # create transformation matrix for N_QL bits
    VH = get_Vg("H")
    Ucb1 = np.kron(VH, np.eye(n))
    Ucb2 = np.kron(Ucb1, Ucb1)

    # create the projector of 0
    Pp = np.zeros((2 * n, 2 * n))
    Id = np.eye(n)

    Pp[0:n, 0:n] = Id
    Pp[n:2*n, 0:n] = Id
    Pp[0:n, n:2*n] = Id
    Pp[n:2*n, n:2*n] = Id
    Pp *= 0.5

    # create the projector of 1
    Pm = Pp.copy()
    Pm[n:2*n, 0:n] *= -1
    Pm[0:n, n:2*n] *= -1

    V0 = get_Vg(gate0, theta, U)
    V1 = get_Vg(gate1, theta, U)

    #U0 = np.kron(V0, Id)
    #U1 = np.kron(V1, Id)

    #UCN = np.linalg.inv(Ucb2) @ (np.kron(Pp, U0) + np.kron(Pm, U1)) @ Ucb2
    U0 = transform1(gate0, n, theta=None, U=None)
    U1 = transform1(gate1, n, theta=None, U=None)
    UCN = (np.kron(Pp, U0) + np.kron(Pm, U1))

    return UCN

def cnot(n, theta=None, U=None):
    """
    Standalone script for CNOT gate matrix for QL-bits
    Parameters:
    n: number of nodes in QL-bit subgraph
    theta: arg for arbitray angle, set to None
    U : Matrix to compare to, set to None here
    """
    UI = transform1('I', n, theta=None,U = None)
    UX = transform1('x', n, theta=None,U = None)
    UZ = transform1('z', n, theta=None,U = None)
    Pp = UI+UZ
    Pm = UI-UZ
    cnot = 0.5*(np.kron(Pp,UI)+np.kron(Pm,UX))
    return cnot


def getRzzgate(n,NQL,theta1):
    """
    Generates Rzz gate matrix for QL-bits
    Parameters:
    n: number of nodes in QL-bit subgraph
    NQL: number of QL-bits
    theta1: arg for arbitray angle
    """
    #Get UCN
    UCN = transform2('I', 'x', n, theta=None, U=None)
    #get Rz
    Rz = transform1('Rz', n, theta=theta1,U = None)
    #get UI
    UI = transform1('I', n, theta=None,U = None)
    URz = np.kron(UI,Rz)
    #URz = np.kron(np.identity((2*n)**(NQL-1)),Rz)
    URzz = UCN @ URz @ UCN
    return URzz

def getRyygate(n,NQL,theta1):
    """
    Generates Ryy gate matrix for QL-bits
    Parameters:
    n: number of nodes in QL-bit subgraph
    NQL: number of QL-bits
    theta1: arg for arbitray angle
    """
    URzz = getRzzgate(n,NQL,theta1)
    #Get UCN
    Rx_p = transform1('Rx', n, theta=1.57,U = None)
    Rx_m = transform1('Rx', n, theta=-1.57,U = None)
    URx_p = np.kron(Rx_p,Rx_p)
    URx_m = np.kron(Rx_m,Rx_m)
    URyy = URx_p @ URzz @ URx_m
    return URzz

def getRxxgate(n,NQL,theta1):
    """
    Generates Rxx gate matrix for QL-bits
    Parameters:
    n: number of nodes in QL-bit subgraph
    NQL: number of QL-bits
    theta1: arg for arbitray angle
    """
    URzz = getRzzgate(n,NQL,theta1)
    #get the Hadamard gate
    VH = transform1('H', n, theta=None,U = None)
    UH = np.kron(VH,VH)
    URxx = UH @ URzz @UH
    return URzz
