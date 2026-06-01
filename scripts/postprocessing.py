import yaml
import argparse
import numpy as np
import igraph as ig
import networkx as nx
from dataclasses import asdict
from qlgates.config import Config
from qlgates.run_dynamics import propagate_state, build_unitary
from qlgates.qlgraphs import qldit, cart_qldit

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

    # Load final states
    psit = np.load(f"./della_slurm_runs/h{cfg.h}_J{cfg.J}_ql_N{cfg.NQL}.npz")['arr_0']
    print(psit.shape)

    #calculate expectation values
    #expectation = expectation(psit, qlbit1_qlbit2)
    
    #plot figures
    
if __name__ == "__main__":
    main()
