import yaml
import argparse
import numpy as np
import igraph as ig
import networkx as nx
from qlgates.config import Config
from qlgates.run_dynamics import propagate_state, build_unitary
from qlgates.qlgraphs import qldit, cart_qldit
from core.graph_generation import generate_quantum_like_bit
from core.contraction import minimal_quotient

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

    # Run your simulation
    #Create QL-resources -- QL-bits, Cartesian Products
    qlbit_1 = qldit(cfg.n, cfg.k, cfg.l, cfg.d, cfg.coupling, cfg.periodic, cfg.full)
    qlbit_2 = qldit(cfg.n, cfg.k, cfg.lp, cfg.d, cfg.coupling, cfg.periodic, cfg.full)
    qlbit1_qlbit2 = cart_qldit(qlbit_1,qlbit_2)
    
    #Encode initial state 
    #e,v = np.linalg.eigh(qlbit1_qlbit2)
    psi0 = computational_basis_state(cfg.NQL, 0)
    
    #Propagate
    psit = propagate_state(cfg,psi0,build_unitary)
    
    #calculate expectation values
    expectation = expectation(psit, qlbit1_qlbit2)
    #plot figures
    
    
    #Checking Will's package
    qlbit_1p, info = generate_quantum_like_bit(cfg.n,cfg.k,cfg.l)


    # print(np.linalg.norm(qlbit_1-qlbit_1p,'fro'))
    # print(np.linalg.norm(qlbit_1-qlbit_2,'fro'))
    # print(np.linalg.norm(qlbit_2-qlbit_1p,'fro'))

    qlbit_1pmin,info_min = minimal_quotient((qlbit_1p,info))
    print(info_min)

if __name__ == "__main__":
    main()
