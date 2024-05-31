import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
import os
from sklearn import metrics
import scipy as sp
from scipy.spatial import distance_matrix

import pdb

from diffusion_functions import *
from semi_supervised_manifold_learning import unweighted_degree

"""
(HYPER)GRAPH CONSTRUCTION

Methods for building neighborhood hyperedges
"""


def build_nbd_hypergraph(base_graph):
    # Want same number of vertices as graph
    n = len(base_graph.nodes())
    # Each vertex contributes a unique hyperedge
    m = n
    # make sure to include vertex v in its own neighborhood hyperedge
    neighborhoods = [[v] + list(base_graph.neighbors(v)) for v in base_graph.nodes()]
    hypergraph = [tuple(edge) for edge in neighborhoods]

    # the 'node dict' is the trivial one
    node_dict = dict(zip(np.arange(n), np.arange(n)))

    # node_dict, labels, label_names
    return dict(
        {
            "n": n,
            "m": m,
            "degree": unweighted_degree(n, hypergraph),
            "hypergraph": hypergraph,
            "node_dict": node_dict,
            "labels": None,
            "label_names": None,
        }
    )


"""
(HYPER)GRAPH CONDUCTANCE

Evaluate conductance of sweepcuts in a (hyper)graph
"""


def add_sweep_cuts_to_dicts(
    x,
    n,
    m,
    D,
    hypergraph,
    phi_by_k_dict,
    best_cut_dict,
    weights=None,
):
    """Find the best sweepcut"""
    if weights is None:
        weights = defaultdict(lambda: 1)
    total_volume = sum(D)
    hyperedges = [list() for _ in range(n)]
    for i, h in enumerate(hypergraph):
        for v in h:
            hyperedges[v].append(i)
    # Make compatible with x.shape= (n,1)
    order = np.argsort(x.flatten())
    is_in_L = np.zeros(n, bool)
    fx = 0
    vol = 0
    S = list()
    for i, v in enumerate(order[:-1]):
        conductance = 0
        S.append(v)
        vol += D[v]
        for h in hyperedges[v]:
            hyperedge_nodes = hypergraph[h]
            h_nodes_in_S = is_in_L[list(hyperedge_nodes)].sum()
            if h_nodes_in_S == 0:
                fx += weights[hyperedge_nodes]
            elif h_nodes_in_S == len(hyperedge_nodes) - 1:
                fx -= weights[hyperedge_nodes]
        is_in_L[v] = True
        conductance = fx / min(vol, total_volume - vol)
        # save to dicts
        k = i + 1
        # if we've found a k-cut of minimal conductance, save cut
        if (not phi_by_k_dict[k]) or (conductance < min(phi_by_k_dict[k])):
            best_cut_dict[k] = S
        phi_by_k_dict[k].append(conductance)
    return phi_by_k_dict, best_cut_dict


"""
SAMPLE CUTS

Sample cuts via diffusion steps
"""


def sample_cuts(
    hgraph_dict: dict,
    initial_nodelist: list,
    phi_by_k_dict: dict,
    best_cut_dict: dict,
    s_vector=None,
    hypergraph_objective=diffusion_functions["infinity"],
    step_size=1,
    num_iterations=100,
    verbose=True,
    hypergraph_node_weights=None,
):
    # let's extract some parameters
    n = hgraph_dict["n"]
    m = hgraph_dict["m"]
    hypergraph = hgraph_dict["hypergraph"]

    degree_dict = hgraph_dict["degree"]
    D = np.array([degree_dict[v] for v in range(n)])

    # create an initial pt corresponding to initial_nodelist
    x0 = np.full(shape=(n, 1), fill_value=0)
    x0[initial_nodelist] = 1
    if s_vector is None:
        s_vector = np.zeros_like(x0)

    # for our hypergraph, first specify the edge objective function
    t, x, y, fx = diffusion(
        x0,
        n,
        m,
        D,
        hypergraph,
        weights=None,
        func=hypergraph_objective,
        lamda=0,
        s=s_vector,
        h=step_size,
        T=num_iterations,
        verbose=verbose,
        hypergraph_node_weights=hypergraph_node_weights,
    )

    def add_all_sweep_cuts_to_dicts(
        x,
        n,
        m,
        D,
        hypergraph,
        phi_by_k_dict=phi_by_k_dict,
        best_cut_dict=best_cut_dict,
        weights=None,
        verbose=True,
    ):
        for idx in range(x.shape[0]):
            if verbose > 0:
                print(f"iterate = {idx}", end="\r")
            phi_by_k_dict, best_cut_dict = add_sweep_cuts_to_dicts(
                x[idx, :],
                n,
                m,
                D,
                hypergraph,
                phi_by_k_dict=phi_by_k_dict,
                best_cut_dict=best_cut_dict,
                weights=None,
            )
        if verbose > 0:
            print()
        return phi_by_k_dict, best_cut_dict

    phi_by_k_dict, best_cut_dict = add_all_sweep_cuts_to_dicts(
        x,
        n,
        m,
        D,
        hypergraph,
        phi_by_k_dict=phi_by_k_dict,
        best_cut_dict=best_cut_dict,
        weights=None,
    )

    return phi_by_k_dict, best_cut_dict


def sample_cuts_from_all_nodes(hgraph_dict):
    n = hgraph_dict['n']
    observed_phi_by_k = dict([(k,list()) for k in range(1,n)])
    best_cut_by_k = dict([(k,list()) for k in range(1,n)])

    for v in list(hgraph_dict['degree'].keys()):
        observed_phi_by_k, best_cut_by_k = sample_cuts(
            hgraph_dict,
            initial_nodelist = [v],
            phi_by_k_dict = observed_phi_by_k, 
            best_cut_dict = best_cut_by_k,
            s_vector=None,
            hypergraph_objective=diffusion_functions["infinity"],
            step_size=1,
            num_iterations=100,
            verbose=True,
            hypergraph_node_weights=None,
        )
    return observed_phi_by_k, best_cut_by_k