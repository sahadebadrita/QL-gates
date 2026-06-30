import yaml
import argparse
import numpy as np
import igraph as ig
import networkx as nx
from scipy.linalg import expm
from qlgates.constants import *
from dataclasses import asdict
from qlgates.config import Config
from qlgates.run_dynamics import propagate_state, build_unitary
from qlgates.qlgraphs import qldit, cart_qldit
from qlgates.cldyn import transverse_field_ising, initial_state_z_up,evolve_times, propagate_state_classical, transverse_ising_trotter
from qlgates.vislib import simpleplot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    """Load YAML and merge into Config dataclass."""
    with open(args.config) as f:
        overrides = yaml.safe_load(f)  # plain dict from YAML

    # These are computed in __post_init__, never let YAML set them
    overrides.pop("l", None)
    overrides.pop("lp", None)

    # Create Config instance with merged parameters
    cfg = Config(**overrides)

    print("=" * 40)
    print("Simulation parameters")
    print("=" * 40)

    for key, value in asdict(cfg).items():
        print(f"  {key}: {value}")
    print("=" * 40)
    Trotter = True
    times = np.arange(0, cfg.timesteps * cfg.deltat, cfg.deltat)
    
    #H,U = exact_unitary(cfg.NQL, cfg.J, 0.0, cfg.deltat) Delete this later
    H = transverse_field_ising(cfg.NQL, cfg.J, cfg.h)
    U = expm(-1j * H * cfg.deltat)
    print("Exact unitary built",flush=True)
    print(H.real)
    print(np.linalg.norm((H-H.T.conj()),'fro')) # Check if H is Hermitian
    
    e_H, v_H = np.linalg.eig(H)
    print("Exact Hamiltonian diagonalized",flush=True)
    psi0 = initial_state_z_up(cfg.NQL)
    psi0_eig = v_H[:,0]/np.linalg.norm(v_H[:,0]) # Assuming psi0 is the first eigenvector (adjust if needed)

    print('Initial state difference from ground state of H(h)',flush=True)
    print('difference b/w psi0 and psi0_eig:', np.linalg.norm(psi0 - psi0_eig))
    print(psi0)
    print(psi0_eig)
    
    
if __name__ == "__main__":
    main()