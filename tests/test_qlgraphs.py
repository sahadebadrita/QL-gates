import numpy as np
import os, sys
import pytest

from qlgates.qlgraphs import qldit, cart_qldit   # adjust import

def test_qldit_shape(small_config):

    R = qldit(
        n=small_config.n,
        k=small_config.k,
        d=small_config.d,
        l=small_config.l,
        coupling=small_config.coupling,
        periodic=small_config.periodic,
        full=small_config.full,
    )

    assert R.shape == (2*small_config.n, 2*small_config.n)

def test_cartpdt_shape(small_config):
    A = np.eye(2*small_config.n)
    B = np.eye(2*small_config.n)

    C = cart_qldit(adj_mat1=A, adj_mat2=B)

    assert C.shape == ((2*small_config.n)**small_config.NQL, (2*small_config.n)**small_config.NQL)
    assert C.ndim == 2

def test_cart_qldit_eigen_convolution(small_config):
    A = qldit(
        n=small_config.n,
        k=small_config.k,
        d=small_config.d,
        l=small_config.l,
        coupling=small_config.coupling,
        periodic=small_config.periodic,
        full=small_config.full,
    )
    B = qldit(
        n=small_config.n,
        k=small_config.k,
        d=small_config.d,
        l=small_config.l,
        coupling=small_config.coupling,
        periodic=small_config.periodic,
        full=small_config.full,
    )
    #A = qldit(n=3, k=2, d=2, l=1, coupling=0.3,
    #          periodic=True, full=False)

    #B = qldit(n=2, k=2, d=2, l=1, coupling=0.3,
    #          periodic=True, full=False)

    C = cart_qldit(adj_mat1=A, adj_mat2=B)

    lamA = np.linalg.eigvals(A)
    lamB = np.linalg.eigvals(B)
    lamC = np.linalg.eigvals(C)

    conv = np.array([a + b for a in lamA for b in lamB])

    assert np.allclose(np.sort(lamC), np.sort(conv), atol=1e-8)
