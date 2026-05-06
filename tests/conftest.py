import numpy as np
import pytest
from qlgates.config import Config

@pytest.fixture
def small_config():
    return Config(
        n=8,
        k=6,
        d=1,
        coupling=True,
        periodic=False,
        full=False,
        NQL=2,
        CartPdt=True
    )

class MockConfig:
    n = 2
    NQL = 2


@pytest.fixture
def cfg():
    return MockConfig()

@pytest.fixture
def psi0(cfg):

    dim = (2 * cfg.n) ** cfg.NQL

    psi = np.ones(dim, dtype=complex)
    psi /= np.linalg.norm(psi)

    return psi
