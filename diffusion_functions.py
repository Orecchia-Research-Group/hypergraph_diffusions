"""
A set of diffusion functions over hypergraphs


"""

from datetime import datetime
from collections import OrderedDict, defaultdict, Counter
import numpy as np
from scipy import sparse


EPS = 1e-6
H = 0.1


# FUCK numpy for not supporting weighted norms
def weighted_median(values, sample_weight=None):
    """ Very close to numpy.percentile, but supports weights.
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

    weighted_quantiles = np.cumsum(sample_weight, axis=0) / np.sum(sample_weight, axis=0)
    return [np.interp(quantiles, weighted_quantiles[:, i], values[i, :]) for i in range(weighted_quantiles.shape[1])]


def compute_hypergraph_matrices(n, m, hypergraph, weights, hypergraph_node_weights=None):
    if weights is None:
        weights = defaultdict(lambda: 1)
    values = []
    i = []
    j = []
    w = []
    for row, e in enumerate(hypergraph):
        values.extend([1] * len(e) if hypergraph_node_weights is None else hypergraph_node_weights[e])
        i.extend([row] * len(e))
        j.extend(e)
        w.append(weights[e])
    W = sparse.diags(w).tocsr()
    sparse_h = sparse.coo_matrix((values, (i, j)), shape=(m, n))
    rank = np.array(sparse_h.sum(axis=0)).squeeze()
    return W, sparse_h, rank


def quadratic(x, s, sparse_h, rank, W, D, center_id=None):
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
    fx = sum([w * wv * np.linalg.norm(x[j] - y[i])**2 for i, j, wv, w in zip(sparse_h.row, sparse_h.col, sparse_h.data, W.data.T)]) / 2 # - np.einsum('ij,ij->', x, s)
    gradient = np.subtract(x, ((sparse_h.T @ W @ y + s).T / D).T)
    return gradient, y, fx


def linear(x, s, sparse_h, rank, W, D, center_id=None):
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
            y[i, :] = weighted_median([x[he[j]] for j in range(k, k+row_counter[i])],
                                      sample_weight=sparse_h.data[k:k+row_counter[i]])
        else:
            y[i, :] = x[center_id[i]]
        k += row_counter[i]
    fx = sum([w * np.linalg.norm(x[j] - y[i], ord=1)**2 for i, j, w in zip(sparse_h.row, sparse_h.col, W.data)]) / 2 # - np.einsum('ij,ij->', x, s)
    gradient = np.subtract(x, ((sparse_h.T @ y + s).T / D).T)
    return gradient, y, fx


def nonvectorized_infinity(x, s, sparse_h, rank, W, D, center_id=None, hypergraph_node_weights=None):
    hypergraph = []
    he = sparse_h.col
    k = 0
    for r in rank:
        hypergraph.append([he[j] for j in range(k, k + int(r))])
        k += int(r)
    gradient = np.zeros(x.shape)
    #degree = np.zeros(x.shape)
    degree = np.array(D)
    if hypergraph_node_weights is None:
        hypergraph_node_weights = {tuple(e): [1] * len(e) for e in hypergraph}
    y = np.zeros((len(rank), x.shape[-1]))
    fx = 0
    for i, e in enumerate(hypergraph):
        xe = x[e]
        we = np.array(hypergraph_node_weights[tuple(e)])
        de = degree[e]
        y_max = xe.max(axis=0)
        y_min = xe.min(axis=0)
        if center_id is None:
            y[i, :] = y_min + (y_max - y_min) / 2
        else:
            y[i, :] = x[center_id[tuple(e)]]
        dist = np.einsum('i,ij->ij', we, (xe - y[i, :]))
        argmax = dist == dist.max(axis=0)
        argmin = dist == dist.min(axis=0)
        # degree[e] += (argmax | argmin) * W[i, i]
        maxmult = argmax.astype(int)
        minmult = argmin.astype(int)
        gradient[e] += W[i, i] * np.einsum('ij,i,ij->ij', dist, de, maxmult / (de @ maxmult) + minmult / (de @ minmult))
        # The following line performs slightly better, but the above line is cleaner
        # and used np.einsum, therefore it is OBVIOUSLY better
        # has not added hyperedge node weights below
        # gradient[e] += W[i, i] * ((xe - y[i, :]).T * de).T * (maxmult / (maxmult.T * de).sum(axis=1) + minmult / (minmult.T * de).sum(axis=1))
        fx += np.linalg.norm(dist, ord=np.inf)
    # degree[degree == 0] = 1
    gradient = (gradient.T / D).T
    gradient -= (s.T / D).T
    # fx -= np.einsum('ij,ij->', x, s)
    return gradient, y, fx


def diffusion(x0, n, m, D, hypergraph, weights, center_id=None, hypergraph_node_weights=None,
              func=nonvectorized_infinity, s=None, h=H, T=None, eps=EPS, verbose=0):

    W, sparse_h, rank = compute_hypergraph_matrices(n, m, hypergraph, weights)

    x = [np.array(x0)]
    if s is None:
        s = np.zeros(shape=x[-1].shape)

    if verbose > 0:
        print(f'Average degree = {sum(D) / n:.3f}. Average rank = {sum(rank) / m:.3f}')
    x = [x0]
    fx = []
    y = []
    crit = 1
    t = 1
    t_start = datetime.now()
    print('{:>10s} {:>6s} {:>13s} {:>14s}'.format('Time (s)', '# Iter', '||dx||_D^2', 'F(x(t))'))
    while (len(x) < 2 or crit > eps) and (T is None or t < T):
        gradient, new_y, new_fx = func(x[-1], s, sparse_h, rank, W, D, center_id=center_id)
        y.append(new_y)
        fx.append(new_fx)
        if verbose > 0:
            t_now = datetime.now()
            print(f'\r{(t_now - t_start).total_seconds():10.3f} {t:6d} {crit:13.6f} {float(fx[-1]):14.6f} {np.abs(gradient).min():10.6f}', end='')
        x.append(x[-1] - h * gradient)
        crit = np.linalg.norm((D * (x[-1] - x[-2]).T) @ (x[-1] - x[-2]))
        t += 1
    _, new_y, new_fx = func(x[-1], s, sparse_h, rank, W, D)
    y.append(new_y)
    fx.append(new_fx)
    if verbose > 0:
        t_now = datetime.now()
        print(f'\r{(t_now - t_start).total_seconds():10.3f} {t:6d} {crit:13.6f} {float(fx[-1]):14.6f}')
    return np.array(x), np.array(y), np.array(fx)


diffusion_functions = OrderedDict([
    ('quadratic', quadratic),
    ('linear', linear),
    ('infinity', nonvectorized_infinity),
])
