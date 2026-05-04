import numpy as np


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
