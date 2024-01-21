"""
Tools for performing semi-supervised clustering by diffusing randomly seeded labels

"""

import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import json
from datetime import datetime
from tqdm import tqdm
from itertools import combinations

from diffusion_functions import *
from animate_diffusion import animate_diffusion

import pdb

"""
DATA GENERATION

Methods for building different "datasets" to cluster on. All methods return two
(2n_pts x 2) numpy arrays: the first is "clean" data, the second has noise added.

All methods construct both arrays such that that the first n/2 columns belong to
community 1 and the latter n/2 columns all belong to community 2.
"""


def generate_spirals(
    tightness=3,
    num_rotations=1.8,
    n_pts=300,
    noise_level=1.2,
    start_theta=np.pi / 2,
    verbose=True,
):
    # generate spiral polar coordinates
    theta = np.sqrt(
        np.linspace(
            start=start_theta**2, stop=(num_rotations * 2 * np.pi) ** 2, num=n_pts
        )
    )
    r = tightness * theta
    # to cartesian coordinates
    spiral_1 = np.vstack([np.multiply(r, np.cos(theta)), np.multiply(r, np.sin(theta))])
    noisy_spiral_1 = spiral_1 + np.random.normal(scale=noise_level, size=(2, n_pts))
    # create second spiral by rotating by angle alpha in the plane
    alpha = np.pi
    rot_mat = np.array(
        [[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]]
    )
    spiral_2 = np.matmul(rot_mat, spiral_1)
    noisy_spiral_2 = spiral_2 + np.random.normal(scale=noise_level, size=(2, n_pts))

    if verbose:
        for man_1, man_2 in [(spiral_1, spiral_2), (noisy_spiral_1, noisy_spiral_2)]:
            plt.plot(man_1[0, :], man_1[1, :], "o", color="r")
            plt.plot(man_2[0, :], man_2[1, :], "o", color="b")
            plt.show()

    # combine into one dataset
    clean_data = np.hstack([spiral_1, spiral_2]).T
    noisy_data = np.hstack([noisy_spiral_1, noisy_spiral_2]).T

    return clean_data, noisy_data


def generate_overlapping_rings(
    r_1=2, r_2=3, n_pts=300, x_shift=3, y_shift=0, noise_level=0.2, verbose=True
):
    theta = np.linspace(start=0, stop=2 * np.pi, num=n_pts)
    ring_1 = np.vstack(
        [np.multiply(r_1, np.cos(theta)), np.multiply(r_1, np.sin(theta))]
    )
    noisy_ring_1 = ring_1 + np.random.normal(scale=noise_level, size=(n_pts, 2)).T

    # I'd like to change the density between the two
    ring_2 = np.vstack(
        [np.multiply(r_2, np.cos(theta)), np.multiply(r_2, np.sin(theta))]
    )
    ring_2 = (
        ring_2
        + np.hstack(
            [
                np.full(shape=(n_pts, 1), fill_value=x_shift),
                np.full(shape=(n_pts, 1), fill_value=y_shift),
            ]
        ).T
    )
    noisy_ring_2 = ring_2 + np.random.normal(scale=noise_level, size=(n_pts, 2)).T

    if verbose:
        for man_1, man_2 in [(ring_1, ring_2), (noisy_ring_1, noisy_ring_2)]:
            plt.plot(man_1[0, :], man_1[1, :], "o", color="r")
            plt.plot(man_2[0, :], man_2[1, :], "o", color="b")
            ax = plt.gca()
            ax.set_aspect("equal")
            plt.show()

    # combine into one dataset
    clean_data = np.hstack([ring_1, ring_2]).T
    noisy_data = np.hstack([noisy_ring_1, noisy_ring_2]).T

    return clean_data, noisy_data


def generate_concentric_highdim(
    ambient_dim=5, r_inner=1, r_outer=2, n_pts=300, noise_level=0.4, verbose=True
):
    outer_shell = np.random.normal(size=(ambient_dim, n_pts))
    # normalize
    outer_shell = r_outer * np.divide(
        outer_shell, np.linalg.norm(outer_shell, ord=2, axis=0)
    )
    noisy_outer_shell = outer_shell + np.random.normal(
        scale=noise_level, size=(ambient_dim, n_pts)
    )

    # inner data
    # random unit vectors
    inner_sphere = np.random.normal(size=(ambient_dim, n_pts))
    inner_sphere = np.divide(inner_sphere, np.linalg.norm(inner_sphere, ord=2, axis=0))
    # sample radii by dim-th root
    radii = r_inner * np.power(
        np.random.uniform(low=0.0, high=1.0, size=n_pts), 1 / ambient_dim
    )
    inner_sphere = np.multiply(radii, inner_sphere)

    # clean_data = inner_sphere.T # np.hstack([outer_shell,inner_sphere]).T
    # noisy_data = inner_sphere.T #np.hstack([noisy_outer_shell,inner_sphere]).T
    clean_data = np.hstack([outer_shell, inner_sphere]).T
    noisy_data = np.hstack([noisy_outer_shell, inner_sphere]).T

    if verbose:
        plot_projection(clean_data, labels="halves")
        plot_projection(noisy_data, labels="halves")

    return clean_data, noisy_data


def plot_projection(high_dim_data, labels=None):
    ax = plt.subplot()
    if labels == "halves":
        community_size = int(high_dim_data.shape[0] / 2)
        plt.plot(
            high_dim_data[:community_size, 0], high_dim_data[:community_size, 1], "o"
        )
        plt.plot(
            high_dim_data[community_size:, 0], high_dim_data[community_size:, 1], "o"
        )
    else:
        plt.plot(high_dim_data[:, 0], high_dim_data[:, 1], "o", c=labels)
    ax.set_aspect("equal")
    plt.show()
    return


"""
(HYPER)GRAPH CONSTRUCTION

Methods for building k-nearest-neighbor graphs and hypergraphs.

Assumes that data_matrix is n x 2, and that the first n/2 rows correspond
 to community 1, second n/2 rows correspond to community 2.
"""


# assumes each node has index [0,n]
def unweighted_degree(n, hypergraph):
    # count unweighted degrees
    degree_dict = dict(zip(np.arange(n), np.zeros(n)))
    for hedge in hypergraph:
        for v in hedge:
            degree_dict[v] += 1
    return degree_dict


def build_knn_hypergraph(data_matrix, k):
    # first neighbor returned is always the node itself, so take k+1 to get k true neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(data_matrix)
    _, indices = nbrs.kneighbors(data_matrix)

    n = data_matrix.shape[0]
    m = indices.shape[0]
    hypergraph = [tuple(edge) for edge in list(indices)]

    # the 'node dict' is the trivial one?
    node_dict = dict(zip(np.arange(n), np.arange(n)))
    # label all pts in spiral 1 as 0, label all pts in spiral 2 as 1
    labels = np.hstack(
        [
            np.full(shape=int(n / 2), fill_value=-1),
            np.full(shape=int(n / 2), fill_value=1),
        ]
    )
    label_names = dict({0: "spiral_1", 1: "sprial_2"})

    # node_dict, labels, label_names
    return dict(
        {
            "n": n,
            "m": m,
            "degree": unweighted_degree(n, hypergraph),
            "hypergraph": hypergraph,
            "node_dict": node_dict,
            "labels": labels,
            "label_names": label_names,
        }
    )


# builds a 2-hypergraph with a hyperedge between a node and each of its k nearest neighbors
def build_knn_hypergraph_star_expansion(data_matrix, k):
    # first neighbor returned is always the node itself, so take k+1 to get k true neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(data_matrix)
    _, indices = nbrs.kneighbors(data_matrix)

    n = data_matrix.shape[0]
    m = k * indices.shape[0]
    star_hypergraph = []
    for hedge_list in list(indices):
        # first entry in each list is the node whose neighbors we're considering
        center = hedge_list[0]
        star_hypergraph.extend([(center, neighbor) for neighbor in hedge_list[1:]])

    # the 'node dict' is the trivial one?
    node_dict = dict(zip(np.arange(n), np.arange(n)))
    # label all pts in spiral 1 as 0, label all pts in spiral 2 as 1
    labels = np.hstack(
        [
            np.full(shape=int(n / 2), fill_value=-1),
            np.full(shape=int(n / 2), fill_value=1),
        ]
    )
    label_names = dict({0: "spiral_1", 1: "sprial_2"})

    # node_dict, labels, label_names
    return dict(
        {
            "n": n,
            "m": m,
            "degree": unweighted_degree(n, star_hypergraph),
            "hypergraph": star_hypergraph,
            "node_dict": node_dict,
            "labels": labels,
            "label_names": label_names,
        }
    )


def build_knn_graph(data_matrix, k):
    star_hgraph_dict = build_knn_hypergraph_star_expansion(data_matrix, k)
    # build networkx graph, updating edge weights when edges ocurr multiple times
    G = nx.Graph()
    for u, v in star_hgraph_dict["hypergraph"]:
        if G.has_edge(u, v):
            G[u][v]["weight"] += 1
        else:
            G.add_edge(u, v, weight=1)
    n = data_matrix.shape[0]
    return nx.adjacency_matrix(G, nodelist=list(range(n)))


# data_matrix has shape (num_nodes x embedding_dimension)
def create_node_weights(method, data_matrix, hgraph_dict):
    # weights_dict is keyed by tuples (i,h) for the weight of node i wrt hyperedge h
    weights_dict = dict()

    # 'gaussian_to_central_neighbor': e^{-||x_i - x_h||} for x_h corresponding to the node whose k-nearest-neighbors for hyperedge h
    if method == "gaussian_to_central_neighbor":
        # assuming that the first index of each hyperedge corresponds to the "central neighbor" of that hyperedge
        assert check_central_neighbor_indices(hgraph_dict["hypergraph"])
        for h_idx, edge in enumerate(hgraph_dict["hypergraph"]):
            central_neighbor = data_matrix[edge[0], :]
            # include "self-weights" on true central neighbor node
            for v in edge:
                x_v = data_matrix[v, :]
                # Optional: normalize according to the dimension of the feature space
                weights_dict[(v, h_idx)] = gaussian_kernel(
                    np.subtract(central_neighbor, x_v), normalize=False
                )
    # centroid is defined as mean of all embedding points in the hyperedge
    elif method == "gaussian_to_centroid":
        for h_idx, edge in enumerate(hgraph_dict["hypergraph"]):
            # get entries of data_matrix corresponding to edge
            edge_embedding = data_matrix[edge, :]
            centroid = np.mean(edge_embedding, axis=0)
            for v in edge:
                x_v = data_matrix[v, :]
                # Optional: normalize according to the dimension of the feature space
                weights_dict[(v, h_idx)] = gaussian_kernel(
                    np.subtract(centroid, x_v), normalize=False
                )
    else:
        raise ValueError(
            f"Unsupported node weight construction method specified: {method}."
        )

    return weights_dict


# Option to normalize by dimension of x: 1/sqrt(sigma^2*(2*pi)^d)*exp(-||x||^2_2/ 2*sigma^2)
def gaussian_kernel(x_vec, sigma=1, normalize=False):
    if normalize:
        dim = x_vec.size
        normalization = np.divide(1, sigma * np.sqrt((2 * np.pi) ** dim))
    else:
        normalization = 1
    return normalization * np.exp(-np.linalg.norm(x_vec, ord=2) ** 2 / (2 * sigma**2))


# iterates through all hyperedges, confirming whether the first entry of the ith hyperedge is node i
def check_central_neighbor_indices(hypergraph):
    central_neighbor_corresponds_to_index = True
    for idx, edge in enumerate(hypergraph):
        central_neighbor_corresponds_to_index = edge[0] == idx
    return central_neighbor_corresponds_to_index


"""
GRAPH DIFFUSION

Methods for vanilla graph diffusion

Currently implemented "slowly". Def possible to speedup for symmetric Laplacians
(i.e. undirected graphs).
"""


def is_symmetric(A, rtol=1e-05, atol=1e-08):
    return np.allclose(A, A.T, rtol=rtol, atol=atol)


def graph_quadratic(L, x):
    return x.T @ L @ x


def eval_graph_cut_fn(D, A, s, x):
    n = A.shape[0]

    D_inv = np.diag(np.divide(1, D))
    L = np.eye(n) - D_inv @ A

    return graph_quadratic(L, x)  # - s@x


def graph_diffusion(x0, D, A, s=None, h=0.5, T=100, verbose=True):
    n = x0.shape[0]
    if np.all(s == None):
        s = np.full(shape=(n, 1), fill_value=0)

    D_inv = np.diag(np.divide(1, D))
    L = np.diag(D) - A
    D_inv_L = np.eye(n) - D_inv @ A

    x = np.reshape(x0, newshape=(1, n))
    y = np.reshape(L @ x0 - s, newshape=(1, n))
    fx = [graph_quadratic(L, x0 - s)]

    x_k = x0
    if verbose:
        t_start = datetime.now()
        print("Starting graph diffusion.")
        print(
            "{:>10s} {:>6s} {:>13s} {:>14s}".format(
                "Time (s)", "# Iter", "||dx||_D^2", "F(x(t))"
            )
        )
    for t in range(T):
        grad = L @ x_k - (D * s.T).T
        # update = D_inv@ grad. Improve stability by implementing D_inv@L as its own operator
        # the current constants in our writeup suggest our hypergraph diffusion is equivalent to
        # performing this graph diffusion with an extra factor of 1/4th on the operator L(x).
        # pdb.set_trace()
        x_k = x_k - h * (D_inv_L @ x_k - D_inv @ s)
        x = np.append(x, np.reshape(x_k, newshape=(1, n)), axis=0)
        y = np.append(y, np.reshape(grad, newshape=(1, n)), axis=0)
        fx.append(graph_quadratic(L, x_k))
        if verbose:
            t_now = datetime.now()
            print(
                f"\r{(t_now - t_start).total_seconds():10.3f} {t:6d} {float(fx[-1]):14.6f} {np.abs(grad).min():10.6f}",
                end="",
            )
    return x, y, fx


"""
SINGLE-TRIAL EXPERIMENTS

Running semi-supervised clustering on a knn (hyper)graph via diffusions versus PPR.

Currently implemented "slowly". Def possible to speedup for symmetric Laplacians
(i.e. undirected graphs).
"""


def eval_hypergraph_cut_fn(
    hypergraph_objective, target_vector, s_vector, sparse_h, rank, W, D
):
    _, _, fx = hypergraph_objective(target_vector, s_vector, sparse_h, rank, W, D)
    return fx


def diffusion_knn_clustering(
    knn_adj_matrix,
    knn_hgraph_dict,
    s_vector=None,
    hypergraph_objective=diffusion_functions["infinity"],
    num_rand_seeds=30,
    step_size=1,
    num_iterations=100,
    verbose=True,
    hypergraph_node_weights=None,
):
    # let's extract some parameters
    n = knn_hgraph_dict["n"]
    m = knn_hgraph_dict["m"]
    hypergraph = knn_hgraph_dict["hypergraph"]

    degree_dict = knn_hgraph_dict["degree"]
    D = np.array([degree_dict[v] for v in range(n)])

    # create an initial pt with num_rand_seeds randomly chosen true labels
    x0 = np.full(shape=(n, 1), fill_value=0)
    random_seeds = np.random.choice(np.arange(n), size=num_rand_seeds)
    x0[random_seeds[random_seeds < n / 2]] = -1
    x0[random_seeds[random_seeds > n / 2]] = 1
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
        s=s_vector,
        h=step_size,
        T=num_iterations,
        verbose=verbose,
        hypergraph_node_weights=None,
    )

    W, sparse_h, rank = compute_hypergraph_matrices(n, m, hypergraph, weights=None)
    hypergraph_cut_objective = lambda vec: eval_hypergraph_cut_fn(
        hypergraph_objective, vec, s_vector, sparse_h, rank, W, D
    )
    hypergraph_diff_results = dict(
        {
            "x": x,
            "y": y,
            "fx": fx,
            "objective": hypergraph_cut_objective,
            "type": "hypergraph",
        }
    )

    # now run the vanilla graph diffusion
    # STEP SIZE 1/2
    x, y, fx = graph_diffusion(
        x0, D, knn_adj_matrix, s=s_vector, h=0.5, T=num_iterations, verbose=verbose
    )

    graph_cut_objective = lambda vec: eval_graph_cut_fn(
        D, knn_adj_matrix, s_vector, vec
    )
    graph_diff_results = dict(
        {"x": x, "y": y, "fx": fx, "objective": graph_cut_objective, "type": "graph"}
    )

    return hypergraph_diff_results, graph_diff_results


def PPR_knn_clustering(
    knn_adj_matrix,
    knn_hgraph_dict,
    error_tolerance=0.1,
    teleportation_factor=0.5,
    hypergraph_objective=diffusion_functions["infinity"],
    num_rand_seeds=30,
    step_size=1,
    num_iterations=100,
    verbose=True,
    hypergraph_node_weights=None,
):
    # teleportation_factor corresponds to a resolvent for lambda = effective_lambda
    effective_lambda = 2 * teleportation_factor / (1 - teleportation_factor)

    # let's extract some parameters
    n = knn_hgraph_dict["n"]
    m = knn_hgraph_dict["m"]
    hypergraph = knn_hgraph_dict["hypergraph"]

    degree_dict = knn_hgraph_dict["degree"]
    D = np.array([degree_dict[v] for v in range(n)])

    # create an s vector proportionate to label vector, with num_rand_seeds randomly chosen true labels
    seeded_labels = np.full(shape=(n, 1), fill_value=0)
    random_seeds = np.random.choice(np.arange(n), size=num_rand_seeds)
    seeded_labels[random_seeds[random_seeds < n / 2]] = -1
    seeded_labels[random_seeds[random_seeds > n / 2]] = 1
    s_vector = effective_lambda * seeded_labels

    # step size: epsilon/2*u_R
    step_size = error_tolerance / (2 * (1 + effective_lambda))

    # Algorithm 1 specifies initialization at 0
    x0 = np.full(shape=(n, 1), fill_value=0)
    _, x, y, fx = diffusion(
        x0,
        n,
        m,
        D,
        hypergraph,
        weights=None,
        func=hypergraph_objective,
        s=s_vector,
        h=step_size,
        T=num_iterations,
        verbose=verbose,
    )
    x_out = (1 - error_tolerance / 2) * np.sum(x, axis=0).flatten()

    W, sparse_h, rank = compute_hypergraph_matrices(n, m, hypergraph, weights=None)
    hypergraph_cut_objective = lambda vec: eval_hypergraph_cut_fn(
        hypergraph_objective, vec, s_vector, sparse_h, rank, W, D
    )
    hypergraph_PPR_results = dict(
        {"x_out": x_out, "objective": hypergraph_cut_objective, "type": "hypergraph"}
    )

    # Now collect graph PPR vector
    # get graph degrees
    D = np.squeeze(np.asarray(np.sum(knn_adj_matrix, axis=0)))
    D_inv = np.diag(np.divide(1, D))

    graph_PPR = np.linalg.solve(
        a=(1 + effective_lambda) * np.eye(n) - D_inv @ knn_adj_matrix,
        b=np.dot(D_inv, s_vector),
    )
    # flatten n x 1 matrix
    graph_PPR = graph_PPR.reshape(n)

    graph_cut_objective = lambda vec: eval_graph_cut_fn(
        D, knn_adj_matrix, s_vector, vec
    )
    graph_PPR_results = dict(
        {"x_out": graph_PPR, "objective": graph_cut_objective, "type": "graph"}
    )

    return hypergraph_PPR_results, graph_PPR_results


def compare_estimated_labels(
    method,
    generate_data,
    k,
    num_iterations,
    diffusion_step_size=None,
    titlestring=None,
    node_weight_method=None,
):
    # generate new data
    _, data_matrix = generate_data(verbose=False)
    n = data_matrix.shape[1]

    # build graph/hypergraph
    knn_adj_matrix = build_knn_graph(data_matrix, k)
    knn_hgraph_dict = build_knn_hypergraph(data_matrix, k)
    if node_weight_method is not None:
        hypergraph_node_weights = create_node_weights(
            method=node_weight_method,
            data_matrix=data_matrix,
            hgraph_dict=knn_hgraph_dict,
        )

    # run diffusion
    if method == "diffusion":
        hypergraph_diff_results, graph_diff_results = diffusion_knn_clustering(
            knn_adj_matrix,
            knn_hgraph_dict,
            num_iterations=num_iterations,
            verbose=False,
            hypergraph_node_weights=hypergraph_node_weights,
        )
        hypergraph_x = hypergraph_diff_results["x"]
        graph_x = graph_diff_results["x"]
        return graph_x[-1, :], hypergraph_x[-1, :], data_matrix

    elif method == "PPR":
        hypergraph_PPR_results, graph_PPR_results = PPR_knn_clustering(
            knn_adj_matrix,
            knn_hgraph_dict,
            error_tolerance=0.1,
            teleportation_factor=0.5,
            num_iterations=num_iterations,
            verbose=False,
            hypergraph_node_weights=None,
        )
        return graph_PPR_results["x_out"], hypergraph_PPR_results["x_out"], data_matrix
    else:
        raise ValueError(
            f'Method should be one of ["diffusion", "PPR"]. Instead got {method}.'
        )


"""
ASSESMENT UTILITIES

Methods for assessing the performance of estimates produced by diffusions.
"""


def make_sweep_cut(vector, threshold):
    mask = np.full(shape=vector.shape, fill_value=np.nan, dtype=int)
    mask[np.where(vector <= threshold)] = -1
    mask[np.where(vector > threshold)] = 1
    return mask


def sweep_cut_classification_error(label_estimates, labels=None):
    n = label_estimates.shape[0]
    if labels is None:
        labels = np.hstack([-np.ones(n // 2, int), np.ones(n - n // 2, int)])
    label_estimates = label_estimates.reshape(-1)
    labels = labels.reshape(-1)
    correct_rate = (label_estimates == labels).sum() / n
    if (len(np.unique(labels)) == 2) and (correct_rate > 1 / 2):
        correct_rate = 1 - correct_rate
    return correct_rate


def find_min_sweepcut(
    node_values,
    resolution,
    cut_objective_function,
    orthogonality_constraint="auto",
    labels=None,
):
    ascending_node_values = sorted(np.unique(node_values))
    # sweep from lowest nontrivial cut to highest nontrivial cut
    low = ascending_node_values[1]
    high = ascending_node_values[-2]

    min_observed_value = np.inf
    best_threshold = low

    if orthogonality_constraint == "auto":
        # find orthogonality constraint created by 0-threshold and add 10% buffer
        zero_estimates = make_sweep_cut(node_values, 0)
        orthogonality_constraint = (
            np.abs(np.sum(zero_estimates) / len(zero_estimates)) + 0.1
        )

    sweep_vals = np.append(np.linspace(low, high, num=resolution, endpoint=False), 0.0)

    for threshold in sweep_vals:
        label_estimates = make_sweep_cut(node_values, threshold)
        objective_value = cut_objective_function(label_estimates, labels=labels)
        orthogonality_error = np.abs(np.sum(label_estimates) / len(label_estimates))

        if (
            objective_value < min_observed_value
            and orthogonality_error < orthogonality_constraint
        ):
            min_observed_value = objective_value
            best_threshold = threshold
    return min_observed_value, best_threshold


def multiclassification_error(label_estimates, labels):
    n = label_estimates.shape[0]
    estimation = np.argmax(label_estimates, axis=1)
    return (estimation != labels).sum() / n


def multiclassification_error_from_x(x, labels):
    """Multi-classification error from x

    Given x, produce the classification error
    for the average iterate of x
    """
    errors = []
    x_cs = np.cumsum(x, axis=0)
    for tt in range(len(x_cs)):
        x_cs[tt] /= tt + 1
        errors.append(multiclassification_error(x_cs[tt], labels))
    return errors


"""
EXPERIMENTS 

Methods for running specific experiments and generating figures.
"""


def small_example_knn_hypergraph():
    # generate new data
    _, data_matrix = generate_spirals(
        n_pts=50, start_theta=np.pi / 5, num_rotations=0.9, verbose=False
    )

    # build a hypergraph from k-nearest-nbs of each point
    k = 5
    knn_hgraph_dict = build_knn_hypergraph(data_matrix, k)
    n = knn_hgraph_dict["n"]
    degree_dict = knn_hgraph_dict["degree"]
    D = np.array([degree_dict[v] for v in range(n)])

    # create an s vector proportionate to label vector, with num_rand_seeds randomly chosen true labels
    num_rand_seeds = int(0.1 * n)
    seeded_labels = np.full(shape=(n, 1), fill_value=0)
    random_seeds = np.random.choice(np.arange(n), size=num_rand_seeds)
    seeded_labels[random_seeds[random_seeds < n / 2]] = -1
    seeded_labels[random_seeds[random_seeds > n / 2]] = 1

    return data_matrix, knn_hgraph_dict, D, seeded_labels


def visualize_labels(method="PPR"):
    k = 5
    target_iternum = 50
    titlestring = "blah"

    _, ax_binary = plt.subplots(nrows=3, ncols=2, figsize=(10, 15))
    problem_index = 0
    for data_generation, problem_kind in [
        (generate_spirals, " Spirals"),
        (generate_overlapping_rings, " Rings"),
        (generate_concentric_highdim, " Concentric Highdim"),
    ]:
        graph_x_out, hypergraph_x_out, data_matrix = compare_estimated_labels(
            method,
            data_generation,
            k,
            target_iternum,
            titlestring=None,
            diffusion_step_size=1,
        )

        for idx, (x, titlestring) in enumerate(
            [(graph_x_out, "Graph"), (hypergraph_x_out, "Hypergraph")]
        ):
            if problem_index == 0:
                plot_label_comparison_binary(
                    ax_binary[problem_index, idx], x, data_matrix, titlestring
                )
            else:
                plot_label_comparison_binary(
                    ax_binary[problem_index, idx],
                    x,
                    data_matrix,
                    titlestring="Abridged",
                )
        problem_index += 1

    plt.suptitle(f"Label estimates \n Iteration {target_iternum}", fontsize=15)
    plt.show()


def compare_AUC_curves(method="PPR"):
    k = 5
    num_iterations = 50
    num_trials = 20

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(6, 15))
    axes_idx = 0

    for data_generation, problem_kind in [
        (generate_spirals, "Spirals"),
        (generate_overlapping_rings, "Rings"),
        (generate_concentric_highdim, "Concentric hyperspheres"),
    ]:
        AUC_vals = []
        for trial in range(num_trials):
            # generate new data
            _, data_matrix = data_generation(verbose=False)

            # build graph/hypergraph
            knn_adj_matrix = build_knn_graph(data_matrix, k)
            knn_hgraph_dict = build_knn_hypergraph(data_matrix, k)

            # run diffusion
            if method == "diffusion":
                hypergraph_diff_results, graph_diff_results = diffusion_knn_clustering(
                    knn_adj_matrix,
                    knn_hgraph_dict,
                    num_iterations=num_iterations,
                    verbose=False,
                )
                graph_x = graph_diff_results["x"]
                hypergraph_x = hypergraph_diff_results["x"]

                graph_x_out = graph_x[-1, :]
                hypergraph_x_out = hypergraph_x[-1, :]
            elif method == "PPR":
                hypergraph_diff_results, graph_diff_results = PPR_knn_clustering(
                    knn_adj_matrix,
                    knn_hgraph_dict,
                    error_tolerance=0.1,
                    teleportation_factor=0.5,
                    num_iterations=num_iterations,
                    verbose=False,
                )
                graph_x_out = graph_diff_results["x_out"]
                hypergraph_x_out = hypergraph_diff_results["x_out"]

            n = data_matrix.shape[0]
            labels = np.hstack(
                [
                    np.full(shape=int(n / 2), fill_value=-1),
                    np.full(shape=int(n / 2), fill_value=1),
                ]
            )
            graph_auc_score = metrics.roc_auc_score(labels, graph_x_out)
            hypergraph_auc_score = metrics.roc_auc_score(labels, hypergraph_x_out)

            AUC_vals.append((hypergraph_auc_score, graph_auc_score))
        titlestring = f"AUC Values at Iteration {num_iterations} \n Results from {num_trials} Independent Trials"
        final_plot_AUC_hist(
            AUC_vals,
            ax=ax[axes_idx],
            decorated=(axes_idx == 0),
            titlestring=titlestring,
        )
        axes_idx += 1
    plt.show()


"""
PLOTTING UTILITIES

Specific visualizations for figures in paper.
"""


def plot_label_comparison_binary(
    ax, label_vector, data_matrix, titlestring=None, labels=None
):
    colors = np.array(["tab:red", "tab:blue", "orange"])
    sweep_cut_resolution = 100
    error, threshold = find_min_sweepcut(
        label_vector,
        sweep_cut_resolution,
        sweep_cut_classification_error,
        labels=labels,
    )
    label_estimates = make_sweep_cut(label_vector, threshold)
    error = sweep_cut_classification_error(label_estimates, labels=labels)
    im = ax.scatter(data_matrix[:, 0], data_matrix[:, 1], c=colors[label_estimates])

    # figure formatting
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.axis("off")
    if titlestring == "Abridged":
        ax.set_title(f"Classification error = {error:.3f}", fontsize=15)
    elif titlestring is not None:
        ax.set_title(
            titlestring + f"\n Classification error = {error:.3f}", fontsize=15
        )
    return error


def final_plot_AUC_hist(AUC_vals, ax, decorated=False, titlestring=None):
    plt.rcParams.update({"font.size": 15})

    hypergraph_vals = [v[0] for v in AUC_vals]
    graph_vals = [v[1] for v in AUC_vals]

    full_values = hypergraph_vals + graph_vals
    _, first_bins = np.histogram(full_values, bins=10)

    # second style
    ax.hist(graph_vals, bins=first_bins, alpha=0.5, edgecolor="black", label="graph")
    ax.hist(
        hypergraph_vals,
        bins=first_bins,
        alpha=0.5,
        edgecolor="black",
        label="hypergraph",
    )

    if decorated:
        if titlestring is not None:
            ax.set_title(titlestring)
        ax.legend()

    # figure formatting
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    return


def visualize_hyperedges(data_matrix, hypergraph):
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.scatter(data_matrix[:, 0], data_matrix[:, 1])
    color_cycle = ax._get_lines.prop_cycler
    for hedge in hypergraph:
        color = next(color_cycle)["color"]
        pairs = list(combinations(hedge, 2))
        for pair in pairs:
            plt.plot(
                data_matrix[pair, 0], data_matrix[pair, 1], linestyle="--", color=color
            )
    plt.show()
    return


# compare_estimated_labels(method='PPR', generate_data = generate_spirals, k=5, num_iterations = 10)

# compare_AUC_curves(method='PPR')
# visualize_labels(method='diffusion')
