"""
Unit tests for cldyn.py functions.
"""
import numpy as np
import pytest
from qlgates.cldyn import initial_state_z_up, transverse_field_ising, transverse_ising_trotter, exact_unitary, propagate_state_classical
from qlgates.helpers import kron_power, computational_basis_state
from qlgates.config import Config
from qlgates.run_dynamics import propagate_state

@pytest.mark.parametrize("NQL", [2,3,4])
def test_unitarity(NQL):
    """U† U should be identity to numerical precision."""
    Ug = transverse_ising_trotter(NQL, J=-1.0, h=0.5, deltat=0.1)
    dim = 2 ** NQL
    err = np.max(np.abs(Ug.conj().T @ Ug - np.eye(dim)))
    assert err < 1e-10, f"Not unitary for NQL={NQL}: max error = {err:.2e}"

@pytest.mark.parametrize("NQL", [2,3,4])
def test_correct_shape(NQL):
    """Output must be (2**NQL, 2**NQL)."""
    Ug = transverse_ising_trotter(NQL, J=-1.0, h=0.5, deltat=0.1)
    assert Ug.shape == (2 ** NQL, 2 ** NQL)

@pytest.mark.parametrize("NQL", [2,3,4])
def test_trotter_converges_to_exact(NQL):
    """
    As dt -> 0, the Trotter unitary should converge to exp(-iHdt).
    Use a small dt and check that the two are close.
    Second-order Trotter error is O(dt^3), so at dt=0.01 we expect ~1e-6.
    """
    J, h, dt = -1.0, 0.5, 0.01
    Ug     = transverse_ising_trotter(NQL, J, h, dt)
    Uexact = exact_unitary(NQL, J, h, dt)
    err = np.max(np.abs(Ug - Uexact))
    assert err < 1e-5, f"Trotter doesn't match exact for NQL={NQL}: max error = {err:.2e}"

@pytest.mark.parametrize("NQL", [2,3,4])
def test_identity_at_zero_dt(NQL):
    """At dt=0 the unitary must be the identity regardless of J, h."""
    Ug  = transverse_ising_trotter(NQL, J=-1.0, h=0.5, deltat=0.0)
    dim = 2 ** NQL
    err = np.max(np.abs(Ug - np.eye(dim)))
    assert err < 1e-10, f"Not identity at dt=0 for NQL={NQL}: max error = {err:.2e}"

@pytest.mark.parametrize("NQL", [2,3,4])
def test_h_zero_is_diagonal(NQL):
    """
    With h=0 the Hamiltonian is purely ZZ (diagonal in computational basis),
    so U must also be diagonal.
    """
    Ug = transverse_ising_trotter(NQL, J=-1.0, h=0.0, deltat=0.3)
    off_diag_err = np.max(np.abs(Ug - np.diag(np.diag(Ug))))
    assert off_diag_err < 1e-10, \
        f"h=0 unitary not diagonal for NQL={NQL}: max off-diag = {off_diag_err:.2e}"

def test_j_zero_factorizes():
    """
    With J=0 the Hamiltonian is a sum of single-site X terms, so
    U must factorize as Rx ⊗ Rx ⊗ ... = kron_power(Rx(2h*dt), NQL).
    """
    NQL, h, dt = 4, 0.7, 0.2
    Ug = transverse_ising_trotter(NQL, J=0.0, h=h, deltat=dt)

    c = np.cos(h * dt);  s = np.sin(h * dt)
    Rx_full = np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)
    U_expected = kron_power(Rx_full, NQL)

    err = np.max(np.abs(Ug - U_expected))
    assert err < 1e-10, f"J=0 unitary doesn't factorize: max error = {err:.2e}"

def test_output_shape(small_config):
    """psit must have shape (2**NQL, timesteps)."""
    psi = computational_basis_state(small_config.NQL, 0)
    psit = propagate_state_classical(small_config, psi)
    assert psit.shape == (2 ** small_config.NQL, small_config.timesteps)

def test_initial_state_preserved(small_config):
    """First column of psit must equal the input psi exactly."""
    psi = computational_basis_state(small_config.NQL, 0)
    psit = propagate_state_classical(small_config, psi)
    np.testing.assert_array_equal(psit[:, 0], psi)

@pytest.mark.parametrize("NQL", [2, 3, 4])
def test_norm_preservation(NQL,small_config):
    """
    Unitary evolution must preserve the norm at every time step.
    Any non-unit norm signals a bug in Ug or the matrix multiply loop.
    """
    small_config.NQL = NQL
    psi = computational_basis_state(NQL, 0)
    psit = propagate_state_classical(small_config, psi)
    norms = np.linalg.norm(psit, axis=0)          # norm at each time step
    np.testing.assert_allclose(norms, 1.0, atol=1e-10,
        err_msg=f"Norm not preserved for NQL={small_config.NQL}")

def test_single_timestep_matches_unitary_applied_once(small_config):
    """
    psit[:,1] must equal Ug @ psi exactly —
    checks that the loop applies Ug once per step, not twice or zero times.
    """
    psi = computational_basis_state(small_config.NQL, 1)
    psit = propagate_state_classical(small_config, psi)

    Ug = transverse_ising_trotter(small_config.NQL, small_config.J, small_config.h, small_config.deltat)
    expected = Ug @ psi
    np.testing.assert_allclose(psit[:, 1], expected, atol=1e-12)

def test_two_timesteps_matches_unitary_applied_twice(small_config):
    """
    psit[:,2] must equal Ug @ Ug @ psi —
    verifies the recurrence psit[:,k] = Ug @ psit[:,k-1] is applied correctly.
    """
    psi = computational_basis_state(small_config.NQL, 0)
    psit = propagate_state_classical(small_config, psi)

    Ug = transverse_ising_trotter(small_config.NQL, small_config.J, small_config.h, small_config.deltat)
    expected = Ug @ Ug @ psi
    np.testing.assert_allclose(psit[:, 2], expected, atol=1e-12)

def test_output_dtype_is_complex(small_config):
    """Output array must be complex even for a real initial state."""
    psi = computational_basis_state(small_config.NQL, 0)        # real-valued
    psit = propagate_state_classical(small_config, psi)
    assert np.iscomplexobj(psit), "psit should be complex dtype"

def test_transverse_field_Ising(small_config):
    """Hamiltonian must have correct shape."""
    H = transverse_field_ising(small_config.NQL, small_config.J, small_config.h)
    assert H.shape == (2 ** small_config.NQL, 2 ** small_config.NQL)

def test_Hamiltonian_hermitian(small_config):
    """Hamiltonian must be Hermitian."""
    H = transverse_field_ising(small_config.NQL, small_config.J, small_config.h)
    assert np.allclose(H, H.conj().T)

def test_Hamiltonian_real(small_config):
    """Hamiltonian must be real-valued."""
    H = transverse_field_ising(small_config.NQL, small_config.J, small_config.h)
    assert np.allclose(H.imag, 0.0)

def test_psi0_shape(small_config):
    """Initial state must have correct shape."""
    psi0 = initial_state_z_up(small_config.NQL)
    assert psi0.shape == (2 ** small_config.NQL,)

def test_psi0_normalized(small_config):
    """Initial state must be normalized."""
    psi0 = initial_state_z_up(small_config.NQL)
    norm = np.linalg.norm(psi0)
    assert np.isclose(norm, 1.0)