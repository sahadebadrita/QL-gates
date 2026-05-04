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
