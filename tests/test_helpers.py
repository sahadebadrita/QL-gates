"""
Unit tests for helper functions in qlgates.helpers.
"""
import numpy as np
import pytest
from qlgates.helpers import kron_power, computational_basis_state, kron_list, build_local_operators
from qlgates.helpers import expectation, build_observable

@pytest.fixture
def identity_2x2():
    return np.eye(2)

@pytest.fixture
def identity_3x3():
    return np.eye(3)

def test_single_identity_unchanged(identity_2x2):
    result = kron_list([identity_2x2])
    np.testing.assert_array_equal(result, identity_2x2)

def test_single_identity_unchanged_3x3(identity_3x3):
    result = kron_list([identity_3x3])
    np.testing.assert_array_equal(result, identity_3x3)

def test_kron_power_identity(identity_2x2):
    result = kron_power(identity_2x2, 3)
    expected = np.eye(8)  # 2^3 x 2^3 identity
    np.testing.assert_array_equal(result, expected)

def test_computational_basis_state():
    NQL = 3
    index = 5
    result = computational_basis_state(NQL, index)
    expected = np.zeros(2 ** NQL, dtype=complex)
    expected[index] = 1.0
    np.testing.assert_array_equal(result, expected)