import yaml
import argparse
import numpy as np
import igraph as ig
import networkx as nx
from dataclasses import asdict
from qlgates.config import Config
from qlgates.run_dynamics import propagate_state, build_unitary
from qlgates.qlgraphs import qldit, cart_qldit
from qlgates.helpers import build_observable, expectation
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

    #calculate expectation values for local Z observable on the first site
    mz = []
    for j in range(cfg.NQL):
        mz.append(build_observable(cfg,'z',j))
    M_eq = []  # Store equilibrium values for each site
    ratios = []
    expectation_mz = np.zeros((cfg.timesteps+1, cfg.NQL+2))  # Store expectation values for each site and time step
    for h in [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]:
        # Load final states
        psit = np.load(f"./della_slurm_runs/h{h}_J{cfg.J}_ql_N{cfg.NQL}.npz")['arr_0']
        for j in range(len(mz)):
            expectation_mz[:,j] = expectation(psit, mz[j])
        expectation_mz[:,-1] = np.arange(cfg.timesteps+1)  # Add time steps as the last column
        expectation_mz[:,-2] = np.mean(expectation_mz[:,:-2], axis=1)  # Add mean expectation values as the second-to-last column
        #save expectation values
        np.savez(f"./della_slurm_runs/h{h}_J{cfg.J}_ql_N{cfg.NQL}_expectation.npz",expectation_mz)
        M_eq.append(np.mean(np.abs(expectation_mz[int(0.8*cfg.timesteps):,-2])))  # Mean of the absolute values of the mean expectation values in the last 20% of time steps
        ratios.append(abs(h/cfg.J))  # Ratio of the equilibrium value at this h to the equilibrium value at the smallest h
    Meq = np.array(M_eq)
    simpleplot(ratios,Meq,
    filename=f"./della_slurm_runs/h{h}_J{cfg.J}_ql_N{cfg.NQL}_magnetization.png",
    xlabel="h/J",
    ylabel="Equilibrium Magnetization",
    title=fr"$\delta t = {cfg.dt},\; T = {cfg.timesteps * cfg.dt}\,\mathrm{fs}$"
)
if __name__ == "__main__":
    main()
