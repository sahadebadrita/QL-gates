import yaml
import argparse
import numpy as np
import igraph as ig
import networkx as nx
from qlgates.constants import *
from dataclasses import asdict
from qlgates.config import Config
from qlgates.helpers import build_local_operators,expectation
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
    psi0 = initial_state_z_up(cfg.NQL)

    #calculate local Z observable on the first site
    for j in range(cfg.NQL):
        mz = build_local_operators(Z,I2,cfg.NQL)
    print('Local observables created',flush=True)

    expectation_mz = np.zeros((cfg.timesteps, cfg.NQL+2))  # Store expectation values for each site and time step
    M_eq = []  # Store equilibrium values for each site
    ratios = []
    
    # Run your simulation
    #for h in [0.2]:
    for h in [0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]:    
        print(f"Running simulation for h={h}...")
        if Trotter:
            Ug = transverse_ising_trotter(cfg.NQL, cfg.J, h, cfg.deltat)
            psit = propagate_state_classical(cfg, psi0, Ug)
        else:
            H = transverse_field_ising(cfg.NQL, h, cfg.J)
            psit = evolve_times(H, psi0, times)
        np.savez(f"./h{h}_J{cfg.J}_cl_N{cfg.NQL}.npz",psit)
    
        #Calculate expectation values
        for j in range(len(mz)):
            expectation_mz[:,j] = expectation(psit, mz[j])
        expectation_mz[:,-1] = np.arange(cfg.timesteps)  # Add time steps as the last column
        expectation_mz[:,-2] = np.mean(expectation_mz[:,:-2], axis=1)  # Add mean expectation values as the second-to-last column
        #save expectation values
        np.savez(f"./h{h}_J{cfg.J}_cl_N{cfg.NQL}_expectation.npz",expectation_mz)
        M_eq.append(np.mean(np.abs(expectation_mz[int(0.0*cfg.timesteps):,-2])))  # Mean of the absolute values of the mean expectation values in the last 20% of time steps
        ratios.append(abs(h/cfg.J))  # Ratio of the equilibrium value at this h to the equilibrium value at the smallest h
    Meq = np.array(M_eq)
    simpleplot(ratios,Meq,
    filename=f"./T{cfg.timesteps * cfg.deltat}_dt{cfg.deltat}_J{cfg.J}_cl_N{cfg.NQL}_magnetization.png",
    xlabel="h/J",
    ylabel="Magnetization (time-averaged)",
    title=fr"$\delta t = {cfg.deltat},\; T = {cfg.timesteps * cfg.deltat}\,\mathrm{{fs}}$")

if __name__ == "__main__":
    main()