import numpy as np
import scipy

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
