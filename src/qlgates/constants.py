import numpy as np

# =============================================================================
# Physical constants (SI units unless noted)
# =============================================================================

hbar   = 1.0545718e-34   # J·s  (reduced Planck constant)
kb     = 1.380649e-23    # J/K  (Boltzmann constant)
ev     = 1.602176634e-19 # J    (electron volt)

# Common energy conversions
mev_to_J  = ev * 1e-3
J_to_mev  = 1.0 / mev_to_J

# Time conversions
fs_to_s   = 1e-15
ps_to_s   = 1e-12
s_to_fs   = 1/fs_to_s

# =============================================================================
# Pauli matrices  (2x2, complex)
# =============================================================================

I2 = np.eye(2, dtype=complex)

X  = np.array([[0, 1],
               [1, 0]], dtype=complex)

Y  = np.array([[ 0, -1j],
               [1j,   0]], dtype=complex)

Z  = np.array([[1,  0],
               [0, -1]], dtype=complex)

# Raising / lowering operators
Sp = np.array([[0, 1],
               [0, 0]], dtype=complex)   # σ+ = (X + iY) / 2

Sm = np.array([[0, 0],
               [1, 0]], dtype=complex)   # σ- = (X - iY) / 2

# Pauli vector  — convenient for dot products: PAULI @ n_hat
PAULI = np.array([X, Y, Z])             # shape (3, 2, 2)

# =============================================================================
# Computational basis states  (column vectors as 1D arrays)
# =============================================================================

ket0 = np.array([1, 0], dtype=complex)  # |0>  spin up
ket1 = np.array([0, 1], dtype=complex)  # |1>  spin down

ket_plus  = np.array([1,  1], dtype=complex) / np.sqrt(2)  # |+> = (|0>+|1>)/√2
ket_minus = np.array([1, -1], dtype=complex) / np.sqrt(2)  # |-> = (|0>-|1>)/√2