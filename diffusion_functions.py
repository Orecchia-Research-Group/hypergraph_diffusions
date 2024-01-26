"""
A set of diffusion functions over hypergraphs


"""

from datetime import datetime
from collections import OrderedDict, defaultdict, Counter
from functools import partial
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import bicgstab, LinearOperator

import pdb


EPS = 1e-8
H = 0.1


def degree_regularizer(D, gradient):
    """Reguralizer is the square matrix D, but we pass tr(D) as a list."""
    return (gradient.T / D).T


def clique_regularizer(L, gradient):
    """The clique expansion should closely resemble the other cut functions"""
    result = np.zeros_like(gradient)
    for i in range(gradient.shape[1]):
        res, info = bicgstab(L, gradient[:, i])
        # print(info)
        result[:, i] = res
    return result


def make_degree_regularizer(n, m, D, hypergraph, weights):
    print("Using degree regularizer")
    return partial(degree_regularizer, D), sparse.diags(D)


def make_clique_regularizer(n, m, D, hypergraph, weights):
    print("Using clique regularizer")
    clique_weights = defaultdict(float)
    for e in hypergraph:
        w = weights[e]
        h = len(e)
        for i in e:
            clique_weights[(i, i)] += w
            for j in e:
                clique_weights[(i, j)] -= w / h
    for i, j in clique_weights.keys():
        if i == j:
            clique_weights[(i, i)] += EPS
    data = []
    row = []
    col = []
    for (i, j), w in clique_weights.items():
        data.append(w)
        row.append(i)
        col.append(j)
    L = sparse.coo_matrix((data, (row, col)), shape=(n, n))

    def mult(v):
        return L @ v + 1000 * v.sum(axis=0)

    operator = LinearOperator((n, n), matvec=mult)
    return partial(clique_regularizer, operator), L


regularizers = {
    "degree": make_degree_regularizer,
    "clique": make_clique_regularizer,
}


def make_regularizer(reg_string, n, m, D, hypergraph, weights):
    """Given a regularizer description and the hypergraph create a preconditioner function"""
    if weights is None:
        weights = defaultdict(lambda: 1)
    return regularizers[reg_string](n, m, D, hypergraph, weights)


# FUCK numpy for not supporting weighted norms
def weighted_median(values, sample_weight=None):
    """Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param sample_weight: array-like of the same length as `array`
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(0.5)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)

    sorter = np.argsort(values, axis=0)
    values = np.array([values[s, i] for i, s in enumerate(sorter.T)])
    sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight, axis=0) / np.sum(
        sample_weight, axis=0
    )
    return [
        np.interp(quantiles, weighted_quantiles[:, i], values[i, :])
        for i in range(weighted_quantiles.shape[1])
    ]


def compute_hypergraph_matrices(
    n, m, hypergraph, weights, hypergraph_node_weights=None
):
    if weights is None:
        weights = defaultdict(lambda: 1)
    values = []
    i = []
    j = []
    w = []
    for row, e in enumerate(hypergraph):
        values.extend(
            [1] * len(e)
            if hypergraph_node_weights is None
            else hypergraph_node_weights[e]
        )
        i.extend([row] * len(e))
        j.extend(e)
        w.append(weights[e])
    W = sparse.diags(w).tocsr()
    sparse_h = sparse.coo_matrix((values, (i, j)), shape=(m, n))
    rank = np.array(sparse_h.sum(axis=1)).squeeze()
    return W, sparse_h, rank


def quadratic(x, sparse_h, rank, W, D, center_id=None):
    """
    Quadratic diffusion

    \\bar{\\delta}_h(x) = \\sum_{i \\in h} |x_i - \\bar{x_h}|

    The diffusion is

    \\partial \\bar{\\delta}_h(x) = \\sign(x_h  - \\bar{x_h})
    """
    if center_id is None:
        y = np.divide((sparse_h @ x).T, rank).T
    else:
        y = x[center_id]
    fx = (
        sum(
            [
                W[i, i] * wv * np.linalg.norm(x[j] - y[i], axis=0) ** 2
                for i, j, wv in zip(sparse_h.row, sparse_h.col, sparse_h.data)
            ]
        )
        / 2
    )  # - np.einsum('ij,ij->', x, s)
    gradient = sparse_h.T @ W @ y  # - (D * x.T).T
    return gradient, y, fx


def linear(x, sparse_h, rank, W, D, center_id=None):
    """
    Linear diffusion

    \\bar{\\delta}_h(x) = \\sum_{i \\in h} \\x_i - median(\\bar{x_h})|

    The diffusion is

    \\partial \\bar{\\delta}_h(x) = \\sign(x_h - median(x_h))
    """
    y = np.zeros([len(rank), x.shape[-1]])
    he = sparse_h.col
    row_counter = Counter(sparse_h.row)
    k = 0
    for i, r in enumerate(rank):
        if center_id is None:
            y[i, :] = weighted_median(
                [x[he[j]] for j in range(k, k + row_counter[i])],
                sample_weight=sparse_h.data[k : k + row_counter[i]],
            )
        else:
            y[i, :] = x[center_id[tuple(he[k : k + row_counter[i]])]]
        k += row_counter[i]
    # print((x.T @ D).sum())
    fx = (
        sum(
            [
                W[i, i] * wv * np.linalg.norm(x[j] - y[i], ord=1, axis=0) ** 2
                for i, j, wv in zip(sparse_h.row, sparse_h.col, sparse_h.data)
            ]
        )
        / 2
    )  # - np.einsum('ij,ij->', x, s)
    gradient = sparse_h.T @ W @ y  # - (D * x.T).T
    return gradient, y, fx


def nonvectorized_infinity(
    x, sparse_h, rank, W, D, center_id=None, hypergraph_node_weights=None
):
    hypergraph = []
    he = sparse_h.col
    k = 0
    for r in rank:
        hypergraph.append([he[j] for j in range(k, k + int(r))])
        k += int(r)
    gradient = np.zeros(x.shape)
    degree = np.array(D)
    if hypergraph_node_weights is None:
        hypergraph_node_weights = {tuple(e): [1] * len(e) for e in hypergraph}
    y = np.zeros((len(rank), x.shape[-1]))
    fx = np.zeros(x.shape[-1])
    for i, e in enumerate(hypergraph):
        if len(e) == 0:
            continue
        xe = x[e]
        we = np.array(hypergraph_node_weights[tuple(e)])
        de = degree[e]
        # Find correct y, using weighted values of x.
        # Weight x using elementwise multipication
        weighted_xe = np.multiply(np.reshape(we, newshape=xe.shape), xe)
        y_max = weighted_xe.max(axis=0)
        y_min = weighted_xe.min(axis=0)
        if center_id is None:
            y[i, :] = y_min + (y_max - y_min) / 2
        else:
            y[i, :] = x[center_id[tuple(e)]]
        # NOTE: why multiply by an extra copy of we below? I've edited to line 230.
        # dist = np.einsum("v,vd->vd", we, (xe - y[i, :]))
        dist = weighted_xe - y[i, :]
        argmax = dist == dist.max(axis=0)
        argmin = dist == dist.min(axis=0)
        # degree[e] += (argmax | argmin) * W[i, i]
        maxmult = argmax.astype(int)
        minmult = argmin.astype(int)
        # normalizes gradient entries by degree of node
        # NOTE: THIS  is where we're currently "dropping" a factor of 1/2, compared to the theoretical
        # diffusion we wrote down. Hence, we recover the same evolution as the graph, without needing
        # to renormalize.
        # NOTE: is the use of degree here to compute a convex combination over argmins and argmaxes?
        # Gradient with node weights has an additional W^T_h out front
        gradient[e] += (
            W[i, i]
            * np.diag(we)
            @ np.einsum(
                "vd,v,vd->vd",
                dist,
                de,
                maxmult / (de @ maxmult) + minmult / (de @ minmult),
            )
        )
        # The following line performs slightly better, but the above line is cleaner
        # and used np.einsum, therefore it is OBVIOUSLY better
        # has not added hyperedge node weights below
        # gradient[e] += W[i, i] * ((xe - y[i, :]).T * de).T * (maxmult / (maxmult.T * de).sum(axis=1) + minmult / (minmult.T * de).sum(axis=1))
        fx += np.linalg.norm(dist, ord=np.inf, axis=0) ** 2
    # gradient = (gradient.T / D).T
    # degree[degree == 0] = 1
    # fx -= np.einsum('ij,ij->', x, s)
    return gradient, y, fx


def added_terms(x, gradient, fx, D, s, f, beta):
    gradient -= f
    fx -= (x * f).sum(axis=0)
    if beta != 0:
        gradient *= 1 - beta
        gradient += beta * (D @ (x - s))
        fx *= 1 - beta
        fx += beta * ((D @ (x - s)) * (x - s)).sum(axis=0) / 2
    return gradient, fx


def diffusion(
    x0,
    n,
    m,
    D,
    hypergraph,
    weights,
    s=None,
    lamda=1,
    center_id=None,
    hypergraph_node_weights=None,
    func=nonvectorized_infinity,
    h=H,
    T=None,
    eps=EPS,
    regularizer="degree",
    verbose=0,
):
    if lamda <= 0:
        print("Warning: lamda <= 0. This will ignore the provided labels.")
    W, sparse_h, rank = compute_hypergraph_matrices(n, m, hypergraph, weights)
    # x[0] -= (D * x[0].T).T.sum(axis=0) / sum(D)
    # x[0] -= (D * x[0].T).T.sum(axis=0) / sum(D)
    # D_mat = sparse.diags(D)

    # Figure out regularizer
    precond_func, R = make_regularizer(regularizer, n, m, D, hypergraph, weights)

    if verbose > 0:
        print(f"Average degree = {sum(D) / n:.3f}. Average rank = {sum(rank) / m:.3f}")
    if s is None:
        s = np.zeros_like(x0)
    x = [x0]
    fx = []
    y = []
    crit = 1
    t = 1
    t_start = datetime.now()
    iteration_times = [0]
    if verbose > 0:
        print(
            "{:>10s} {:>6s} {:>13s} {:>14s}".format(
                "Time (s)", "# Iter", "||dx||_D^2", "F(x(t))"
            )
        )
    while True:
        # \nabla f(x) = \lambda D x - s + \sum_h w_h \bar{\delta}_h(x) \partial \bar{delta}_h(x)
        gradient, new_y, new_fx = func(
            x[-1],
            sparse_h,
            rank,
            W,
            D,
            center_id=center_id,
            hypergraph_node_weights=hypergraph_node_weights,
        )
        disagreement = x[-1] - s
        gradient += lamda * (D * x[-1].T).T - s
        new_fx += lamda * ((D * x[-1].T).T * x[-1]).sum(axis=0) / 2 - (x[-1] * s).sum(
            axis=0
        )
        y.append(new_y)
        fx.append(new_fx)
        iteration_times.append((datetime.now() - t_start).total_seconds())
        if verbose > 0:
            print(
                f"\r{iteration_times[-1]:10.3f} {t:6d} {crit:13.6f} {float(fx[-1].min()):14.6f}",
                end="",
            )
        # NOTE: The degree regularizer precond_func multiplies by D_inv, but multiplying gradient
        # by a copy of D up in line 303 seems to counteract this?
        # NOTE: it seems that ADDITIONAL degree regularization is taking place in line 233? Or is that computing
        # a convex combination of the argmins/maxes? If additional regularization is taking place, is that why
        # the diffusion is still operating correctly despite the above "self-negating" degree regularization?
        # dx = -h * precond_func(gradient)
        new_x = x[-1] - h * precond_func(gradient)
        # print(np.array([x[-1].reshape(-1), cut_gradient.reshape(-1), label_gradient.reshape(-1), gradient.reshape(-1), dx.reshape(-1)]))
        # input()
        if (len(x) > 2 and crit <= eps) or (T is not None and t >= T):
            if verbose > 0:
                t_now = datetime.now()
                print(
                    f"\r{(t_now - t_start).total_seconds():10.3f} {t:6d} {crit:13.6f} {float(fx[-1].min()):14.6f}"
                )
            break
        x.append(new_x)
        crit = ((R @ (x[-1] - x[-2])) * (x[-1] - x[-2])).sum(axis=0).max() / 2
        t += 1

    return np.array(iteration_times), np.array(x), np.array(y), np.array(fx)


def sweep_cut(
    x, n, m, D, hypergraph, weights=None, center_id=None, hypergraph_node_weights=None
):
    """Find the best sweepcut"""
    if weights is None:
        weights = defaultdict(lambda: 1)
    total_volume = sum(D)
    hyperedges = [list() for _ in range(n)]
    for i, h in enumerate(hypergraph):
        for v in h:
            hyperedges[v].append(i)
    order = np.argsort(x)
    is_in_L = np.zeros(n, bool)
    fx = 0
    vol = 0
    value = np.zeros(n - 1)
    volume = np.zeros(n - 1)
    conductance = np.zeros(n - 1)
    for i, v in enumerate(order[:-1]):
        vol += D[v]
        for h in hyperedges[v]:
            hyperedge_nodes = hypergraph[h]
            h_nodes_in_L = is_in_L[list(hyperedge_nodes)].sum()
            if h_nodes_in_L == 0:
                fx += weights[hyperedge_nodes]
            elif h_nodes_in_L == len(hyperedge_nodes) - 1:
                fx -= weights[hyperedge_nodes]
        is_in_L[v] = True
        value[i] = fx
        volume[i] = vol
        conductance[i] = fx / min(vol, total_volume - vol)
    return value, volume, conductance


def all_sweep_cuts(x, n, m, node_weights, hypergraph, verbose):
    value = np.zeros((x.shape[1], x.shape[0] - 1))
    volume = np.zeros(value.shape)
    conductance = np.zeros(value.shape)
    for d, xtd in enumerate(x.T):
        if verbose > 0:
            print(f"d = {d:3d}", end="\r")
        value[d], volume[d], conductance[d] = sweep_cut(
            xtd, n, m, node_weights, hypergraph
        )
    if verbose > 0:
        print()
    return value, volume, conductance


diffusion_functions = OrderedDict(
    [
        ("quadratic", quadratic),
        ("linear", linear),
        ("infinity", nonvectorized_infinity),
    ]
)
