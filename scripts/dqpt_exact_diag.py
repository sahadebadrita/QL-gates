"""
Exact diagonalization of the transverse field Ising model for DQPT.

Hamiltonian:  H = -J * sum_{<i,j>} sz_i sz_j  +  h * sum_i sx_i
                  (1D chain, periodic boundary conditions)

Ground state at h0=0, J=+1 is the ferromagnetic |00...0> state.
Quench: prepare ground state of H(J, h0), then evolve under H(J, h1).
Compute Loschmidt amplitude G(t) = <psi(0)|psi(t)> and return rate lambda(t).
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


# ── Pauli matrices ────────────────────────────────────────────────────────────

sx = sp.csr_matrix(np.array([[0, 1], [1, 0]], dtype=complex))
sz = sp.csr_matrix(np.array([[1, 0], [0, -1]], dtype=complex))


def kron_op(op, i, N):
    """Place 'op' on site i of an N-site chain (rest are identity)."""
    ops = [sp.eye(2, format="csr")] * N
    ops[i] = op
    result = ops[0]
    for o in ops[1:]:
        result = sp.kron(result, o, format="csr")
    return result


# ── Hamiltonian builder ───────────────────────────────────────────────────────

def build_hamiltonian(N, J, h, periodic=True):
    """
    H = -J * sum_{<i,j>} sz_i sz_j  +  h * sum_i sx_i
    1D chain with periodic boundary conditions by default.
    """
    dim = 2 ** N
    H = sp.csr_matrix((dim, dim), dtype=complex)

    # ZZ interaction
    n_bonds = N if periodic else N - 1
    for i in range(n_bonds):
        j = (i + 1) % N
        H += -J * kron_op(sz, i, N) @ kron_op(sz, j, N)

    # Transverse field
    for i in range(N):
        H += h * kron_op(sx, i, N)

    return H


# ── Ground state ──────────────────────────────────────────────────────────────

def ground_state(H):
    """Return the ground state (lowest eigenvalue) of H."""
    dim = H.shape[0]
    if dim <= 4:
        # Dense for tiny systems
        vals, vecs = np.linalg.eigh(H.toarray())
        return vals[0], vecs[:, 0]
    else:
        vals, vecs = spla.eigsh(H, k=1, which="SA")
        return vals[0], vecs[:, 0]


# ── Time evolution via matrix exponentiation ──────────────────────────────────

def evolve(H, psi0, t):
    """
    Exact time evolution: |psi(t)> = exp(-i H t) |psi0>
    Uses full diagonalization — exact but O(2^3N) cost.
    Fine for N <= 14 or so.
    """
    H_dense = H.toarray()
    evals, evecs = np.linalg.eigh(H_dense)          # H = V D V†
    # coefficients in energy eigenbasis
    c = evecs.conj().T @ psi0                        # c_n = <n|psi0>
    phase = np.exp(-1j * evals * t)                  # e^{-i E_n t}
    return evecs @ (phase * c)                       # sum_n c_n e^{-iEnt} |n>


# ── Loschmidt quantities ──────────────────────────────────────────────────────

def loschmidt_rate(N, J, h0, h1, t_array, periodic=True):
    """
    Returns lambda(t) = -(1/N) ln |<psi(0)|psi(t)>|^2

    psi(0) = ground state of H(J, h0)
    psi(t) = exp(-i H1 t) |psi(0)>  where H1 = H(J, h1)
    """
    H0 = build_hamiltonian(N, J, h0, periodic)
    H1 = build_hamiltonian(N, J, h1, periodic)

    _, psi0 = ground_state(H0)

    # Pre-diagonalise H1 once, reuse for all t
    H1_dense = H1.toarray()
    evals, evecs = np.linalg.eigh(H1_dense)
    c = evecs.conj().T @ psi0          # overlap in energy eigenbasis

    lam = np.zeros(len(t_array))
    for k, t in enumerate(t_array):
        phase = np.exp(-1j * evals * t)
        psi_t = evecs @ (phase * c)
        G = np.vdot(psi0, psi_t)       # <psi(0)|psi(t)>  (vdot conjugates left arg)
        L = np.abs(G) ** 2             # Loschmidt echo
        lam[k] = -np.log(max(L, 1e-300)) / N

    return lam


# ── Analytical result (thermodynamic limit, free fermions) ────────────────────

def loschmidt_rate_analytical(J, h0, h1, t_array, n_k=2000):
    """
    Thermodynamic limit via free-fermion solution.
    Valid for the ferromagnetic case J>0 with h crossing hc=J.
    k-sum over momentum modes k = (2n+1)pi/N, n=0,...,N/2-1.
    """
    ks = np.array([(2*n + 1) * np.pi / n_k for n in range(n_k)])

    def eps(k, h):
        return 2 * np.sqrt(J**2 + h**2 - 2*J*h*np.cos(k))

    e0 = eps(ks, h0) / 2
    e1 = eps(ks, h1) / 2

    # Bogoliubov angle between pre- and post-quench vacua
    num = h0*h1 + J**2 - J*(h0 + h1)*np.cos(ks)
    cos_theta = num / (e0 * e1)
    cos_theta = np.clip(cos_theta, -1, 1)
    sin2_theta = 1 - cos_theta**2

    lam = np.zeros(len(t_array))
    for k, t in enumerate(t_array):
        A = cos_theta * np.cos(e1 * t)
        B = np.sqrt(np.maximum(sin2_theta, 0)) * np.sin(e1 * t)
        log_fk = np.log(np.maximum(A**2 + B**2, 1e-300))
        lam[k] = -np.mean(log_fk)

    return lam


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Parameters — change these freely
    J  = 1.0    # +1: ferromagnetic; ground state at h0=0 is |00...0>
    h0 = 0.0    # pre-quench field
    h1 = 2.0    # post-quench field  (crosses hc=|J|=1 → DQPTs expected)
    N  = 10     # system size

    t_array = np.linspace(0, 15, 800)

    print(f"Building H0 (N={N}, J={J}, h={h0}) ...")
    print(f"Building H1 (N={N}, J={J}, h={h1}) ...")
    print("Running exact diagonalization ...")

    lam_ed = loschmidt_rate(N, J, h0, h1, t_array)

    print("Computing thermodynamic-limit (analytical) result ...")
    lam_ana = loschmidt_rate_analytical(J, h0, h1, t_array)

    # ── Critical times (thermodynamic limit) ─────────────────────────────────
    # k* is the mode where cos(theta_k*) = 0
    ks_dense = np.linspace(1e-6, np.pi, 10000)
    e0_d = np.sqrt(J**2 + h0**2 - 2*J*h0*np.cos(ks_dense))
    e1_d = np.sqrt(J**2 + h1**2 - 2*J*h1*np.cos(ks_dense))
    num_d = h0*h1 + J**2 - J*(h0 + h1)*np.cos(ks_dense)
    cos_th = num_d / (e0_d * e1_d + 1e-15)
    idx = np.argmin(np.abs(cos_th))
    k_star = ks_dense[idx]
    eps1_star = np.sqrt(J**2 + h1**2 - 2*J*h1*np.cos(k_star))
    t_star = np.pi / (2 * eps1_star)
    t_crits = [(2*n + 1) * t_star for n in range(6) if (2*n+1)*t_star <= t_array[-1]]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    fig.suptitle(
        rf"DQPT — Transverse Field Ising Model  "
        rf"($J={J},\ h_0={h0},\ h_1={h1},\ N={N}$)",
        fontsize=13
    )

    # Top panel: return rate
    ax = axes[0]
    ax.plot(t_array, lam_ana, lw=1.5, color="#185FA5", label="Analytical (N→∞)", zorder=3)
    ax.plot(t_array, lam_ed,  lw=1.5, color="#D85A30", ls="--", label=f"Exact diag. N={N}", zorder=2)
    for i, tc in enumerate(t_crits):
        ax.axvline(tc, color="grey", lw=0.8, ls=":", alpha=0.7,
                   label=r"$t^*_n$" if i == 0 else None)
    ax.set_ylabel(r"$\lambda(t) = -\frac{1}{N}\ln\mathcal{L}(t)$", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Bottom panel: Loschmidt echo itself
    H0 = build_hamiltonian(N, J, h0)
    H1 = build_hamiltonian(N, J, h1)
    _, psi0 = ground_state(H0)
    H1_dense = H1.toarray()
    evals, evecs = np.linalg.eigh(H1_dense)
    c = evecs.conj().T @ psi0

    echo = np.zeros(len(t_array))
    for k, t in enumerate(t_array):
        psi_t = evecs @ (np.exp(-1j * evals * t) * c)
        echo[k] = np.abs(np.vdot(psi0, psi_t))**2

    ax2 = axes[1]
    ax2.plot(t_array, echo, lw=1.5, color="#D85A30", label=rf"$\mathcal{{L}}(t)$, N={N}")
    for tc in t_crits:
        ax2.axvline(tc, color="grey", lw=0.8, ls=":", alpha=0.7)
    ax2.set_ylabel(r"$\mathcal{L}(t) = |\langle\psi(0)|\psi(t)\rangle|^2$", fontsize=12)
    ax2.set_xlabel("time  $t$", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("./dqpt_exact_diag.png", dpi=150, bbox_inches="tight")
    print("Saved plot to dqpt_exact_diag.png")
    plt.show()

    # ── Print critical times ──────────────────────────────────────────────────
    print(f"\nCritical mode k* = {k_star:.4f}")
    print(f"t* (fundamental)  = {t_star:.4f}")
    print("Critical times t*_n:")
    for i, tc in enumerate(t_crits):
        print(f"  t*_{i+1} = {tc:.4f}")
