import numpy as np
import os, sys
import pytest

from qlgates.qldyn import transverseN, bell_state

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
    psi = bell_state(cfg, psi0, "x")

    assert psi.shape[0] == (2 * cfg.n) ** 2
    assert np.isclose(
        np.vdot(psi, psi),
        1.0
    )

def test_bell_orthogonality(cfg, psi0):
    psi1 = bell_state(cfg, psi0, "I")
    psi2 = bell_state(cfg, psi0, "x")

    overlap = np.vdot(psi1, psi2)

    assert np.isclose(overlap, 0.0)
