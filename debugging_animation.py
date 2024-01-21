import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os

import pdb

from semi_supervised_manifold_learning import (
    diffusion_functions,
    diffusion,
    graph_diffusion,
)
from submodular_cut_fns import submodular_subgradient, cardinality_cut_fn


def animate_hgraph_diffusion(
    data_matrix, hypergraph_dict, x0, T=49, step_size=1, verbose=False
):
    # let's extract some parameters
    n = hypergraph_dict["n"]
    m = hypergraph_dict["m"]
    hypergraph = hypergraph_dict["hypergraph"]
    degree_dict = hypergraph_dict["degree"]
    D = np.array([degree_dict[v] for v in range(n)])
    s_vector = np.zeros_like(x0)

    # for our hypergraph, first specify the edge objective function
    cut_func = diffusion_functions["infinity"]
    t, x, y, fx = diffusion(
        x0,
        n,
        m,
        D,
        hypergraph,
        weights=None,
        func=cut_func,
        s=s_vector,
        h=step_size,
        T=T,
        verbose=verbose,
    )
    successful_iterations = x.shape[0]

    # animate results
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Hypergraph diffusion iteration {0} \n 2 hypergraph")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # pdb.set_trace()
    flat_x = np.reshape(x, newshape=(successful_iterations, n))

    def update(frame):
        color = flat_x[frame, :]
        im = ax.scatter(data_matrix[:, 0], data_matrix[:, 1], c=color)
        ax.set_title(f"Hypergraph diffusion iteration {frame} \n 2 hypergraph")
        cax.cla()
        fig.colorbar(im, cax=cax)
        return im

    ani = animation.FuncAnimation(
        fig=fig, func=update, frames=successful_iterations, interval=300
    )
    plt.close()

    return ani


def animate_graph_diffusion(data_matrix, A, D, x0, T=49, step_size=0.5, verbose=False):
    n = data_matrix.shape[0]

    x, y, fx = graph_diffusion(x0, D, A, s=None, h=step_size, T=T, verbose=verbose)

    # animate results
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(f"Graph diffusion iteration {0} \n update x_k - 0.5 D_inv(L(x)-s)")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    def update(frame):
        color = x[frame, :]
        im = ax.scatter(data_matrix[:, 0], data_matrix[:, 1], c=color)
        ax.set_title(
            f"Graph diffusion iteration {frame}\n update x_k - 0.5 D_inv(L(x)-s)"
        )
        cax.cla()
        fig.colorbar(im, cax=cax)
        return im

    ani = animation.FuncAnimation(fig=fig, func=update, frames=T, interval=300)
    plt.close()

    return ani
