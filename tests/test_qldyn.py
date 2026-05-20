"""
Unit tests for hamiltonians.py and qldyn.py functions.
"""

import numpy as np
import os, sys
import pytest

from qlgates.hamiltonians import transverseN
from qlgates.run_dynamics import bell_state, build_unitary, propagate_state, build_transverse_unitary

def test_transverseN(small_config):
    U = transverseN(
        n=small_config.n,
        NQL=small_config.NQL,
        J=small_config.J,
        h=small_config.h,
        dt=0.1,
        debug=False
            )
    assert U.shape == ((2*small_config.n)**small_config.NQL, (2*small_config.n)**small_config.NQL)

def test_bell_state_norm(small_config, psi0):
    psi = bell_state(small_config, psi0, "phi_plus")
    assert np.isclose(
        np.vdot(psi, psi),
        1.0
    )

def test_bell_orthogonality(small_config, psi0):
    psi1 = bell_state(small_config, psi0, "phi_plus")
    psi2 = bell_state(small_config, psi0, "phi_minus")
    overlap = np.vdot(psi1, psi2)
    assert np.isclose(overlap, 0.0)

def test_bell_state_shape(small_config, psi0):
    psi = bell_state(small_config, psi0, "psi_minus")
    assert psi.shape == psi0.shape

def test_transverseN_is_unitary(small_config):
    """Trotter unitary must be unitary for any parameters."""
    Ug = transverseN(small_config.n, small_config.NQL, small_config.J, small_config.h, small_config.deltat, debug=False)
    should_be_identity = Ug @ Ug.conj().T
    assert np.allclose(should_be_identity, np.eye(len(Ug)), atol=1e-12)

def test_propagate_preserves_norm(small_config, psi0):
    """Norm must be conserved at every time step."""
    psit = propagate_state(small_config, psi0, build_unitary)
    norms = np.linalg.norm(psit, axis=0)
    assert np.allclose(norms, 1.0, atol=1e-10)

def test_norm_preserved(small_config, psi0):
    """Norm must be 1 at every time step (unitary evolution)."""
    psit = propagate_state(small_config, psi0, build_unitary)
    norms = np.linalg.norm(psit, axis=0)
    assert np.allclose(norms, 1.0, atol=1e-10)