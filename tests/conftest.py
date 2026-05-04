import pytest
from qlgates.config import Config

@pytest.fixture
def small_config():
    return Config(
        n=4,
        k=2,
        d=1,
        coupling=True,
        periodic=False,
        full=False,
    )
