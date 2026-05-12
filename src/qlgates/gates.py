import numpy as np

#Scripts that relate to gate transformations -- both single and two qubit

def get_Vg(gate, theta=None, U=None):
    """
    Generate the single qubit gate matrix in computational basis.

    Parameters:
    gate : str - Name of the quantum gate to be applied.
    theta : optional - Angle parameter for parameterized gates (if applicable).
    U : optional - External unitary matrix for generalized transformations.

    Returns:
    Vg : np.ndarray - Conventional single qubit gate matrix.
    """
    Vg = np.zeros((2,2), dtype=complex)

    if gate == "U":
        if U is None:
            raise ValueError("You must provide a 2x2 unitary matrix U for 'Ucustom'.")
        if not isinstance(U, np.ndarray) or U.shape != (2, 2):
            raise ValueError("U must be a 2x2 NumPy array.")
        if not np.allclose(U.conj().T @ U, np.eye(2), atol=1e-10):
            raise ValueError("Provided matrix U is not unitary.")
        return U
    if gate == "I":
        Vg[0,0] = 1.
        Vg[0,1] = 0.
        Vg[1,0] = 0.
        Vg[1,1] = 1.

    elif gate == "H":
        Vg[0,0] = 1.
        Vg[0,1] = 1
        Vg[1,0] = 1
        Vg[1,1] = -1

        Vg /= np.sqrt(2.)

    elif gate == "x":
        Vg[0,0] = 0.
        Vg[0,1] = 1.
        Vg[1,0] = 1.
        Vg[1,1] = 0.

    elif gate == "y":
        Vg[0,0] = 0.
        Vg[0,1] = -1j
        Vg[1,0] = 1j
        Vg[1,1] = 0

    elif gate == "z":
        Vg[0,0] = 1.
        Vg[0,1] = 0
        Vg[1,0] = 0
        Vg[1,1] = -1

    elif gate == "Rz":
        if theta is None:
            raise ValueError("Theta must be provided for Rz gate.")
        Vg[0,0] = np.exp(-1j * theta / 2)
        Vg[1,1] = np.exp(1j * theta / 2)
        Vg[0,1] = 0.
        Vg[1,0] = 0.

    elif gate == "Ry":
        if theta is None:
            raise ValueError("Theta must be provided for Ry gate.")
        Vg[0,0] = np.cos(theta/2)
        Vg[0,1] = -np.sin(theta/2)
        Vg[1,0] = np.sin(theta/2)
        Vg[1,1] = np.cos(theta/2)

    elif gate == "Rx":
        if theta is None:
            raise ValueError("Theta must be provided for Rx gate.")
        Vg[0,0] = np.cos(theta/2)
        Vg[0,1] = -1j * np.sin(theta/2)
        Vg[1,0] = -1j * np.sin(theta/2)
        Vg[1,1] = np.cos(theta/2)

    else:
        raise ValueError("Wrong gate name. Supported gates: I, H, x, y, z, Rz, Ry, Rx, custom U")

    return Vg

def transform1(gate, n, theta=None, U=None):
    """
    Generate the single QL-bit gate matrix in the transformed basis.

    Parameters:
    gate : str - Name of the quantum gate to be applied.
    n : int - Number of nodes in each QL-bit subgraph.
    theta : optional - Angle parameter for parameterized gates (if applicable).
    U : optional - External unitary matrix for generalized transformations.

    Returns:
    Ug : np.ndarray - Transformed single QL-bit gate matrix.
    """

    VH = get_Vg("H")
    Vg = get_Vg(gate, theta, U)

    Ucb = np.kron(VH, np.identity(n))

    Ug = (
        np.linalg.inv(Ucb)
        @ np.kron(Vg, np.identity(n))
        @ Ucb
    )

    return Ug

def transform2(gate0, gate1, n, theta=None, U=None):
    """
    Generate a controlled transformation matrix for QL-bits.

    Parameters:
    gate0 : str - Conditional gate applied to the target when the control state is 0.
    gate1 : str - Conditional gate applied to the target when the control state is 1.
    n : int - Number of nodes in each QL-bit subgraph.
    theta : optional - Angle parameter for parameterized gates (if applicable).
    U : optional - External unitary matrix for generalized transformations.

    Returns:
    UCN : np.ndarray - Controlled transformation matrix for the QL-bit system.
    """

    # create transformation matrix for N_QL bits
    VH = get_Vg("H")
    Ucb1 = np.kron(VH, np.eye(n))
    Ucb2 = np.kron(Ucb1, Ucb1)

    # create the projector of 0
    Pp = np.zeros((2 * n, 2 * n))
    Id = np.eye(n)

    Pp[0:n, 0:n] = Id
    Pp[n:2*n, 0:n] = Id
    Pp[0:n, n:2*n] = Id
    Pp[n:2*n, n:2*n] = Id
    Pp *= 0.5

    # create the projector of 1
    Pm = Pp.copy()
    Pm[n:2*n, 0:n] *= -1
    Pm[0:n, n:2*n] *= -1

    U0 = transform1(gate0, n, theta=None, U=None)
    U1 = transform1(gate1, n, theta=None, U=None)
    UCN = (np.kron(Pp, U0) + np.kron(Pm, U1))

    return UCN

def cnot(n, theta=None, U=None):
    """
    Generate the CNOT gate matrix for QL-bits.

    Parameters:
    n : int - Number of nodes in each QL-bit subgraph.
    theta : optional - Angle parameter (not used in this implementation, kept for compatibility).
    U : optional - External matrix input for comparison or generalization (not used here).
    Returns:
    cnot : np.ndarray - CNOT gate matrix for the QL-bit system.
    """
    UI = transform1('I', n, theta=None,U = None)
    UX = transform1('x', n, theta=None,U = None)
    UZ = transform1('z', n, theta=None,U = None)
    Pp = UI+UZ
    Pm = UI-UZ
    cnot = 0.5*(np.kron(Pp,UI)+np.kron(Pm,UX))
    return cnot


def getRzzgate(n,NQL,theta1):
    """
    Generate the Rzz rotation gate matrix for QL-bits.

    Parameters:
    n : int - Number of nodes in each QL-bit subgraph.
    NQL : int - Number of QL-bits in the system.
    theta1 : float - Rotation angle parameter for the Rzz gate.

    Returns:
    URzz : np.ndarray - Rzz gate matrix for the QL-bit system.
    """
    UCN = transform2('I', 'x', n, theta=None, U=None)
    Rz = transform1('Rz', n, theta=theta1,U=None)
    UI = transform1('I', n, theta=None,U=None)
    URz = np.kron(UI,Rz)
    URzz = UCN @ URz @ UCN
    return URzz

def getRyygate(n,NQL,theta1):
    """
    Generate the Ryy rotation gate matrix for QL-bits.

    Parameters:
    n : int - Number of nodes in each QL-bit subgraph.
    NQL : int - Number of QL-bits in the system.
    theta1 : float - Rotation angle parameter for the Ryy gate.

    Returns:
    URyy : np.ndarray - Ryy gate matrix for the QL-bit system.
    """
    URzz = getRzzgate(n,NQL,theta1)
    Rx_p = transform1('Rx', n, theta=1.57,U=None)
    Rx_m = transform1('Rx', n, theta=-1.57,U=None)
    URx_p = np.kron(Rx_p,Rx_p)
    URx_m = np.kron(Rx_m,Rx_m)
    URyy = URx_p @ URzz @ URx_m
    return URyy

def getRxxgate(n, NQL, theta1):
    """
    Generate the Rxx rotation gate matrix for QL-bits.

    Parameters:
    n : int - Number of nodes in each QL-bit subgraph.
    NQL : int - Number of QL-bits in the system.
    theta1 : float - Rotation angle parameter for the Rxx gate.

    Returns:
    URxx : np.ndarray - Rxx gate matrix for the QL-bit system.
    """
    URzz = getRzzgate(n,NQL,theta1)
    VH = transform1('H', n, theta=None,U = None)
    UH = np.kron(VH,VH)
    URxx = UH @ URzz @UH
    return URxx
