# ============================================================
# Imports
# ============================================================
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import expm, logm
from scipy.integrate import solve_ivp
from scipy.stats import unitary_group
import igraph as ig
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
from scipy import sparse

def plot_normalized_density_of_states(all_eigenvalues, bin_width, top_d,fname,fig_label):
    """
    Plots the normalized density of states (DOS) using eigenvalues from all samples.

    Parameters:
    all_eigenvalues (numpy.ndarray): A 2D NumPy array where each column contains eigenvalues from a sample.
    bin_width (float): Width of each bin for the histogram.
    top_d (int): Number of top eigenvalues to display.
    """

    all_eigenvalues = np.asarray(all_eigenvalues)

    # Reshape 1D input to (N, 1) so it behaves like a single column sample
    if all_eigenvalues.ndim == 1:
        all_eigenvalues = all_eigenvalues[:, np.newaxis]
    num_samples = all_eigenvalues.shape[1]  # Number of columns (samples)

    # Define a common bin range across all samples
    min_eigenvalue, max_eigenvalue = all_eigenvalues.min(), all_eigenvalues.max()

    print("min_eigenvalue:", min_eigenvalue)
    print("max_eigenvalue:", max_eigenvalue)
    print("bin_width:", bin_width)

    if bin_width <= 0 or min_eigenvalue == max_eigenvalue:
        raise ValueError("Invalid bin_width or eigenvalue range")

    bins = np.arange(min_eigenvalue, max_eigenvalue + bin_width, bin_width)

    plt.figure(figsize=(10,8))

    # Loop over columns (samples) and plot each histogram
    for i in range(num_samples):
        plt.hist(
            all_eigenvalues[:, i], bins=bins, density=True, alpha=0.5,linewidth=1,
            edgecolor='#756bb1',color='#bcbddc'
        )
    # ✅ Fix: remove y-axis tick labels (corrected)
    #ax = plt.gca()
    #ax.set_yticklabels([])
    #ax.tick_params(axis='y', labelleft=False)

    # Set y-axis to log scale
    #plt.yscale("log")
    #plt.title("Normalized Density of States")
    plt.xlabel(r"$\lambda$")
    plt.ylabel(r"$\rho(\lambda)$")
    plt.tight_layout()
    plt.yticks([])  # ✅ Removes y-axis ticks and labels completely
    #plt.xticks([-5,0,5,10,15],[-5,0,5,10,15])
    # Add figure number at top-left of the figure (in normalized figure coordinates)
    fig = plt.gcf()
    fig.text(0.09, 0.96, fig_label, ha='left', va='top')
    #plt.savefig(f'plotdos_{fname}.pdf')
    plt.show()


def visgraph(R, n, filename, fig_label):
    """
    Visualizes a block-structured graph with different colors for intra-subgraph 
    (A matrices), inter-subgraph (C matrices), and anti-diagonal edges.

    Parameters:
    R (np.ndarray): The full adjacency matrix of the network.
    n (int): Number of nodes per subgraph.
    """
    # Create graph from adjacency matrix
    G = ig.Graph.Adjacency((R > 0).tolist(), mode=ig.ADJ_UNDIRECTED)

    total_nodes = R.shape[0]  # Total nodes in the system
    d = total_nodes // n  # Number of subgraphs (diagonal blocks)

    layout = []
    outer_radius = 300  # distance between subgraph centers
    jitter_radius = 60  # spread within each cluster

    for block_id in range(d):
        # Cluster center on outer ring
        theta = 2 * np.pi * block_id / d
        cx = outer_radius * np.cos(theta)
        cy = outer_radius * np.sin(theta)

        # Place each node randomly near the cluster center
        for _ in range(n):
            dx, dy = np.random.normal(0, jitter_radius, size=2)
            layout.append((cx + dx, cy + dy))


    # Default colors and alphas
    edge_colors = []
    edge_alphas = []
    edge_tuples = [e.tuple for e in G.es]  # List of (source, target) edges

    # Assign colors and alphas based on block structure
    for edge in edge_tuples:
        u, v = edge
        # Check if (u, v) belong to the same subgraph (A block)
        in_same_block = (u // n) == (v // n)
        # Check if (u, v) belong to the anti-diagonal edges
        in_anti_diagonal = ((u // n) + (v // n)) == (d - 1)
        # Check if (u, v) are between the first and last subgraph
        between_first_last = ((u // n) == 0 and (v // n) == (d - 1)) or ((u // n) == (d - 1) and (v // n) == 0)

        if in_same_block:
            edge_colors.append("#d73027")  # A matrices (diagonal blocks)
            edge_alphas.append(1.0)  # Fully opaque
        elif between_first_last:
            edge_colors.append("#4575b4")  # Edges between first and last subgraph
            edge_alphas.append(1.0)  # Fully transparent
        else:
            edge_colors.append("#4575b4")  # C matrices (off-diagonal blocks)
            edge_alphas.append(1.0)  # Fully opaque

    # Convert edge colors to RGBA with transparency
    edge_colors_rgba = [(r, g, b, alpha) for (r, g, b, a), alpha in zip(plt.cm.colors.to_rgba_array(edge_colors), edge_alphas)]

    # Define visual style
    visual_style = {
        "vertex_size": 6,
        "vertex_color": "gray",
        "vertex_outline_color": "gray",
        "vertex_outline_width": 1,
        "edge_color": edge_colors_rgba,  # Apply custom edge colors with transparency
        "edge_width": 1,
        "edge_curved": False,
        "layout": G.layout("kk"),
        #"layout": layout,
        "bbox": (1000, 1000),
        "margin": 100
    }

    # Plot using matplotlib (no cairo required)
    fig, ax = plt.subplots(figsize=(8, 6))
    ig.plot(
        G,
        target=ax,
        vertex_size=visual_style["vertex_size"],
        vertex_color=visual_style["vertex_color"],
        edge_color=visual_style["edge_color"],  # Apply custom edge colors with transparency
        edge_width=visual_style["edge_width"],
        edge_curved=visual_style["edge_curved"],
        layout=visual_style["layout"]
    )
    # Add figure number at top-left of the figure (in normalized figure coordinates)
    fig.text(0.2, 0.8, fig_label, ha='left', va='top')
    plt.savefig('visgraph_%s.pdf'%(filename))
    plt.show()
    
