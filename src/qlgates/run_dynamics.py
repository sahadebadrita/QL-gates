from qlgates.qldyn import transverseN, xymodel_two_qubit_trotter
from qlgates.gates import transform1, transform2
from qlgates.config import Config
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
"""
Build the unitaries for the transverse field Ising model and the XY model. 
The code is structured to allow easy extension to other models by adding new unitary builders to the registry.
"""
# trotter/transverse.py
def build_transverse_unitary(cfg) -> np.ndarray:
    return transverseN(cfg.n, cfg.NQL, cfg.J, cfg.h, cfg.deltat, debug=False)

# trotter/xy.py
def build_xy_unitary(cfg) -> np.ndarray:
    return xymodel_two_qubit_trotter(cfg.n, cfg.NQL, cfg.J, cfg.deltat)

_UNITARY_REGISTRY = {
    "transverse": build_transverse_unitary,
    "xy":         build_xy_unitary,
}

def build_unitary(cfg) -> np.ndarray:
    """Single entry point. Reads cfg.model and dispatches to the right builder."""
    if cfg.model not in _UNITARY_REGISTRY:
        raise ValueError(
            f"Unknown model '{cfg.model}'. "
            f"Available: {list(_UNITARY_REGISTRY)}"
        )
    return _UNITARY_REGISTRY[cfg.model](cfg)

"""
Create different initial states for propagation using the unitaries.
The code is structured to allow easy extension to other initial states by adding them to the registry.
"""

def bell_state(cfg: Config, psi0, kind="phi_plus"):

    """
    Generate a Bell basis state from the initial |00⟩ state.

    Parameters:
    cfg : object - Configuration object containing system parameters.
    psi0 : np.ndarray - Initial state vector, expected to represent the |00...0⟩ state.
    kind : str - Type of Bell state to generate ("phi_plus", "phi_minus", "psi_plus", or "psi_minus").
        'phi_plus'  -> |Φ⁺⟩
        'phi_minus' -> |Φ⁻⟩
        'psi_plus'  -> |Ψ⁺⟩
        'psi_minus' -> |Ψ⁻⟩

    Returns:
    state : np.ndarray - Generated Bell state vector.
    """

    UH = transform1("H", cfg.n)
    UI = transform1("I", cfg.n)
    CNOT = transform2("I", "x", cfg.n)

    # Bell entangling circuit
    Ubell = CNOT @ np.kron(UH, UI)
    
    # Generate other Bell states via local ops on qubit 0
    ops = {
        "phi_plus": np.kron(transform1("I", cfg.n), UI),
        "phi_minus": np.kron(transform1("z", cfg.n), UI),
        "psi_plus": np.kron(transform1("x", cfg.n), UI),
        "psi_minus": np.kron(transform1("z", cfg.n) @ transform1("x", cfg.n), UI)
    }

    if kind not in ops:
        raise ValueError(f"Unknown Bell state: {kind}")
    Ug = ops[kind] @ Ubell
    psi_bell = Ug @ psi0
    return psi_bell

_INITSTATE_REGISTRY = {
    "all_zero": build_transverse_unitary,
    "bell_state":         bell_state,
}

def build_initstate(cfg) -> np.ndarray:
    """Single entry point. Reads cfg.initstate and dispatches to the right builder."""
    if cfg.initstate not in _INITSTATE_REGISTRY:
        raise ValueError(
            f"Unknown initial state '{cfg.initstate}'. "
            f"Available: {list(_INITSTATE_REGISTRY)}"
        )
    return _INITSTATE_REGISTRY[cfg.initstate](cfg)

def propagate_state(cfg: Config, psi: np.ndarray, build_unitary: callable) -> np.ndarray:
    """
    Propagate an initial quantum state through time using the transverse model evolution operator.

    Parameters:
    cfg : object - Configuration object containing system and simulation parameters.
    psi : np.ndarray - Initial state vector of the system.

    Returns:
    psit : np.ndarray - Time-evolved state vectors for all simulation time steps.
    """
    logging.info('Propagate_state')
    Ntot = (2 * cfg.n) ** cfg.NQL
    #Ug = transverseN(cfg.n, cfg.NQL, cfg.J, cfg.h, cfg.deltat,debug=False)
    Ug = build_unitary(cfg)
    psit = np.empty((Ntot,cfg.timesteps),dtype=complex)
    psit[:,0] = psi
    print(cfg.timesteps)
    for step in range(1,cfg.timesteps,1):
        print('yyyy')
        print("time", step*cfg.deltat,flush=True)
        psit[:,step] = Ug @ psit[:,step-1]

    return psit