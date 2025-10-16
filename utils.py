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

#Scripts for building the quantum like resources and Cartesian Product

def qldit(n, k, l, d, coupling, periodic, full):
    """
    Efficiently generates the adjacency matrix for a QL-dit system.

    Parameters:
    n : int - Number of vertices per subgraph.
    k : int - Degree of regular graphs for diagonal blocks.
    l : int - Degree of coupling graphs for off-diagonal blocks.
    d : int - Number of subgraphs - 1 (determines coupling layers).
    coupling : boolean - True: coupling ON, False: coupling OFF
    periodic : boolean - True: periodic chain ON, False: periodic chain OFF
    full : boolean - True: full coupling ON, False: full coupling OFF
    Returns:
    system_matrix : np.ndarray - The full system adjacency matrix.
    """
    total_nodes = n * (d + 1)
    system_matrix = np.zeros((total_nodes, total_nodes), dtype=np.int8)  # Efficient memory allocation

    # Generate adjacency matrices for k-regular graphs (diagonal blocks)
    adjacency_matrices = [np.array(ig.Graph.K_Regular(n, k).get_adjacency().data, dtype=np.int8) for _ in range(d+1)]
    # Assign adjacency matrices to diagonal blocks
    for i in range(d + 1):
        start, end = i * n, (i + 1) * n
        system_matrix[start:end, start:end] = adjacency_matrices[i]

    if coupling == True:
        # Generate coupling adjacency matrices (l-regular graphs for off-diagonal blocks)
        coupling_matrices = [np.array(ig.Graph.K_Regular(n, l).get_adjacency().data, dtype=np.int8) for _ in range(d)]
        # Assign coupling matrices to superdiagonal and subdiagonal
        for i in range(d):
            start, end = i * n, (i + 1) * n
            start_next, end_next = (i + 1) * n, (i + 2) * n

            system_matrix[start:end, start_next:end_next] = coupling_matrices[i]  # Superdiagonal
            system_matrix[start_next:end_next, start:end] = coupling_matrices[i]  # Subdiagonal (symmetric)
    if periodic == True:
        coupling_matrices.append(np.array(ig.Graph.K_Regular(n, l).get_adjacency().data, dtype=np.int8))
        # Assign coupling matrices to the corners of R
        system_matrix[:n,-n:] = coupling_matrices[-1]  # Superdiagonal
        system_matrix[-n:,:n] = coupling_matrices[-1]  # Subdiagonal (symmetric)
    # Implement all-to-all coupling between subgraphs
    if full == True:
        # Generate a single l-regular coupling matrix
        all_to_all_coupling = np.array(ig.Graph.K_Regular(n, l).get_adjacency().data, dtype=np.int8)

        # Assign coupling matrices to all subgraph pairs
        for i in range(d + 1):
            for j in range(i + 1, d + 1):  # Avoid diagonal blocks
                start_i, end_i = i * n, (i + 1) * n
                start_j, end_j = j * n, (j + 1) * n

                system_matrix[start_i:end_i, start_j:end_j] = all_to_all_coupling
                system_matrix[start_j:end_j, start_i:end_i] = all_to_all_coupling  # Ensure symmetry
    return system_matrix


def cart_qldit(n,d,adj_mat1,adj_mat2):
    """
    Creates the cartesian product of two QL-dits
    Parameters:
    n : number of vertices in a subgraph
    d : dimension of QL-dit
    adj_mat1 : adjacency matrix for QL-dit 1
    adj_mat2 : adjacency matrix for QL-dit 2
    """
    adj_matrix_cart_kron = np.kron(np.eye(int((d+1)*n)),adj_mat2) + np.kron(adj_mat1,np.eye(int((d+1)*n)))
    return adj_matrix_cart_kron

#Scripts that relate to gate transformations -- both single and two qubit

def get_Vg(gate, theta=None, U=None):

	Vg = np.zeros((2,2),dtype=complex)

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
		#sys.exit()

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

#single qubit gate implementation
def transform1(gate, n, theta=None, U=None):

	VH = get_Vg("H")
	Vg = get_Vg(gate,theta,U)
	Ucb = np.kron(VH, np.identity(n))
	Ug =  (np.linalg.inv(Ucb) @
	np.kron(Vg,np.identity(n)) @ Ucb)

	#Rg =  Ug @ R @ Ug.T.conj()

	return Ug

# two qubit conditional gate implementation
def transform2(gate0, gate1, n, theta=None, U=None):
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

#two qubit rotation gate implementation
def transform1(gate, n, NQL, theta=None):
    if gate == "Rxx":
        Ug = getRxxgate(n,NQL,theta)
    elif gate == "Ryy":
        Ug = getRyygate(n,NQL,theta)
    elif gate == "Rzz":
        Ug = getRzzgate(n,NQL,theta)

	return Ug

def is_hermitian(M, label, rtol=1e-05, atol=1e-08):

	if M.shape[0] != M.shape[1]:
		raise ValueError("Operator must be a square matrix.")

	# Check if operator is equal to its conjugate transpose
	result = np.allclose(M, M.conj().T, rtol=rtol, atol=atol)
	if result: print(label, "is Hermitian")
	else: print(label, " is not Hermitian")

	return result

def is_unitary(M, label, rtol=1e-05, atol=1e-08):

	if M.shape[0] != M.shape[1]:
		raise ValueError("Matrix must be square.")

	# Compute the conjugate transpose (Hermitian adjoint)
	M_dagger = M.conj().T

	# Check if matrix_dagger * matrix = I and matrix * matrix_dagger = I
	identity = np.eye(M.shape[0], dtype=M.dtype)

	result = np.allclose(M_dagger @ M, identity, rtol=rtol, atol=atol) and \
	np.allclose(M @ M_dagger, identity, rtol=rtol, atol=atol)

	if result: print(label, "is unitary")
	else: print(label, "is not unitary")

	# Check if operator is equal to its conjugate transpose
	return result

def spectral_decomposition(M, threshold,  text=True, hermitian=False):

	#higher precision: helps with spectral decomposition
	M = M.astype(np.complex128)

	if M.shape[0] != M.shape[1]:
		print("spectral decomposition defined for squared matrices")
		#sys.exit()

	###extract vector of emergent states
	if hermitian: lambda_, v = np.linalg.eigh(M)
	else: lambda_, v = np.linalg.eig(M)

	idx = lambda_.argsort()[::-1]
	lambda_ = lambda_[idx]
	v = v[:,idx]

	rank = np.linalg.matrix_rank(M)

	if rank != M.shape[0]:
		print("rank ", rank, "<", M.shape[0])
		#sys.exit()

	N = len(lambda_)
	Mspec = np.zeros((N,N), dtype=np.complex128)

	for i in range(0,N):
		Mspec += lambda_[i]*np.outer(v[:, i], v[:, i].conj())

	for i in range(0,N):
		for j in range(0,N):
			if abs(M[i,j]-Mspec[i,j])>threshold:
				print("issue in spectral decomposition:", M[i,j], Mspec[i,j])
				sys.exit()

	if text: print("spectral decomposition OK, rank: ", rank)

	return lambda_, v
