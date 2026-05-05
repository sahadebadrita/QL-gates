import numpy as np
import os, sys
import pytest

from qlgates.qldyn import transverseN

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
