from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Config:
    debug: bool = False

    # --- QL graph ---
    n: int = 16
    k: int = 8
    d: int = 1
    l: Optional[int] = None
    lp: Optional[int] = None

    coupling: bool = True
    periodic: bool = False
    full: bool = False

    NQL: int = 2
    CartPdt: bool = False

    def __post_init__(self):
        self.l = int((self.k - 2 * np.sqrt(self.k - 1)) / 2)
        self.lp = self.l - 1

        # Validation
        assert self.k < self.n, f"k={self.k} must be less than n={self.n}"
        assert self.deltat > 0, "deltat must be positive"
        assert self.model in ("transverse", "xy"), f"Unknown model: {self.model}"

    model : str = "transverse" # "transverse" or "xy"
    timesteps : int = 1
    deltat : float = 0.1

    # --- Hamiltonian ---
    J: float = -1.0
    h: float = 2.0
