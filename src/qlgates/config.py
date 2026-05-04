from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class Config:
    debug: bool = False

    # --- QL graph ---
    n: int = 12
    k: int = 8
    d: int = 1
    l: Optional[int] = None
    lp: Optional[int] = None

    coupling: bool = True
    periodic: bool = False
    full: bool = False

    NQL: int = 3
    CartPdt: bool = True

    def __post_init__(self):
        self.l = int((self.k - 2 * np.sqrt(self.k - 1)) / 2)
        self.lp = self.l - 1
