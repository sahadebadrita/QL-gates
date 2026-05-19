import numpy as np
import os, sys
import pytest

from qlgates.hamiltonians import transverseN
from qlgates.run_dynamics import bell_state
from tests.conftest import small_config
#from tests.conftest import small_config

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

def test_bell_state_norm(cfg, psi0):
    psi = bell_state(cfg, psi0, "phi_plus")

    assert np.isclose(
        np.vdot(psi, psi),
        1.0
    )

def test_bell_orthogonality(cfg, psi0):
    psi1 = bell_state(cfg, psi0, "phi_plus")
    psi2 = bell_state(cfg, psi0, "phi_minus")

    overlap = np.vdot(psi1, psi2)

    assert np.isclose(overlap, 0.0)

def test_bell_state_shape(cfg, psi0):

    psi = bell_state(cfg, psi0, "psi_minus")

    assert psi.shape == psi0.shape

def test_transverseN_is_unitary():
    """Trotter unitary must be unitary for any parameters."""
    #cfg = small_config
    Ug = transverseN(small_config.n, small_config.NQL, small_config.J, small_config.h, small_config.deltat, debug=False)
    should_be_identity = Ug @ Ug.conj().T
    assert np.allclose(should_be_identity, np.eye(len(Ug)), atol=1e-10)