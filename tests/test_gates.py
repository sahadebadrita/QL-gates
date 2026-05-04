import numpy as np
import os, sys
import pytest

from qlgates.gates import getRzzgate   # adjust import

#@pytest.mark.parametrize("theta1",[0.2,0.4,0.8])
#@pytest.mark.parametrize("n",[2,4,8])
def test_rzz_shape(small_config):
    U = getRzzgate(small_config.n, NQL=2., theta1=0.5)
    assert U.shape == (4*small_config.n*small_config.n, 4*small_config.n*small_config.n)

def test_rzz_theta_zero_identity():
    n = 2
    U = getRzzgate(n, NQL=2, theta1=0.0)
    I = np.eye(4*n*n)
    assert np.allclose(U, I)
