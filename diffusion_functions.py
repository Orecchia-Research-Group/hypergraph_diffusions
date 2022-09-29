"""
A set of diffusion functions over hypergraphs


"""

from datetime import datetime
from collections import OrderedDict
import numpy as np
from scipy import sparse


EPS = 1e-6
H = 0.1


def diffusion(x0, n, m, D, hypergraph, func, s=None, h=H, T=None, eps=EPS, verbose=0):
    values = []
    i = []
    j = []
    for row, e in enumerate(hypergraph):
        values.extend([1] * len(e))
        i.extend([row] * len(e))
        j.extend(e)
    sparse_h = sparse.coo_matrix((values, (i, j)), shape=(m, n))
    print('Created sparse matrix')
    x = [np.array(x0)]
    if s is None:
        s = np.zeros(shape=x[-1].shape)
    rank = np.array(sparse_h.sum(axis=1)).squeeze()
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
        gradient, new_y, new_fx = func(x[-1], s, sparse_h, rank, D)
        y.append(new_y)
        fx.append(new_fx)
        if verbose > 0:
            t_now = datetime.now()
            print(f'\r{(t_now - t_start).total_seconds():10.3f} {t:6d} {crit:13.6f} {float(fx[-1]):14.6f} {np.abs(gradient).min():10.6f}', end='')
        x.append(x[-1] - h * gradient)
        crit = np.linalg.norm((D * (x[-1] - x[-2]).T) @ (x[-1] - x[-2]))
        t += 1
    _, new_y, new_fx = func(x[-1], s, sparse_h, rank, D)
    y.append(new_y)
    fx.append(new_fx)
    if verbose > 0:
        t_now = datetime.now()
        print(f'\r{(t_now - t_start).total_seconds():10.3f} {t:6d} {crit:13.6f} {float(fx[-1]):14.6f}')
    return np.array(x), np.array(y), np.array(fx)


def quadratic(x, s, sparse_h, rank, D):
    """
    Quadratic diffusion

    \\bar{\\delta}_h(x) = \\sum_{i \\in h} |x_i - \\bar{x_h}|

    The diffusion is

    \\partial \\bar{\\delta}_h(x) = \\sign(x_h  - \\bar{x_h})
    """
    y = np.divide((sparse_h @ x).T, rank).T
    fx = sum([w * np.linalg.norm(x[j] - y[i])**2 for i, j, w in zip(sparse_h.row, sparse_h.col, sparse_h.data)]) / 2 # - np.einsum('ij,ij->', x, s)
    gradient = np.subtract(x, ((sparse_h.T @ y + s).T / D).T)
    return gradient, y, fx


def linear(x, s, sparse_h, rank, D):
    """
    Linear diffusion

    \\bar{\\delta}_h(x) = \\sum_{i \\in h} |x_i - median(\\bar{x_h})|

    The diffusion is

    \\partial \\bar{\\delta}_h(x) = \\sign(x_h - median(x_h))
    """
    y = np.zeros([len(rank), x.shape[-1]])
    he = sparse_h.col
    k = 0
    for i, r in enumerate(rank):
        y[i, :] = np.median([x[he[j]] for j in range(k, k+int(r))], axis=0)
        k += int(r)
    fx = sum([w * np.linalg.norm(x[j] - y[i], ord=1)**2 for i, j, w in zip(sparse_h.row, sparse_h.col, sparse_h.data)]) / 2 # - np.einsum('ij,ij->', x, s)
    gradient = np.subtract(x, ((sparse_h.T @ y + s).T / D).T)
    return gradient, y, fx


# This would be the vectorized version.
# Unfortunately it doesn't seem to work.
def infinity(x, s, sparse_h, rank, D):
    """
    Range diffusion using the infinity norm

    \\bar{\\delta}_h(x) = \\max_{i \\in h} x_i - \\min_{j \\in h} x_j

    The subgradient is

    \\partial \\bar{\\delta}_h(x) = \\mathbbm{1}_{i \\in argmax x} - \\mathbbm{1}_{j \\in argmin x}
    """
    m, n = sparse_h.shape
    he = sparse_h.col
    xe = []
    x_d = []
    ymax = []
    ymin = []
    y = []
    gradient = []
    for d in range(x.shape[-1]):
        xe_values = [x[j, d] * v for i, j, v in zip(sparse_h.row, sparse_h.col, sparse_h.data)]
        xe.append(sparse.coo_matrix((xe_values, (sparse_h.row, sparse_h.col)), shape=(m, n)))
        ymax.append(np.zeros(len(rank)))
        ymin.append(np.zeros(len(rank)))
        k = 0
        for i, r in enumerate(rank):
            ymax[d][i] = np.max([x[he[j], d] for j in range(k, k+int(r))])
            ymin[d][i] = np.min([x[he[j], d] for j in range(k, k+int(r))])
            # if i == 20:
            #     print(ymax[d][i], ymin[d][i])
            #     print([x[he[j], d] for j in range(k, k+int(r))])
            #     input()
            k += int(r)
        y.append(np.array(ymin[d] + (ymax[d] - ymin[d]) / 2).squeeze())
        ymax_values = np.array([ymax[d][i] * v for i, j, v in zip(sparse_h.row, sparse_h.col, sparse_h.data)])
        ymin_values = np.array([ymin[d][i] * v for i, j, v in zip(sparse_h.row, sparse_h.col, sparse_h.data)])
        # y_values = ymin_values + (ymax_values - ymin_values) / 2
        x_argmax_values = (xe_values == ymax_values).astype(int)
        x_argmin_values = (xe_values == ymin_values).astype(int)
        x_d_values = x_argmax_values + x_argmin_values
        x_d_coo = sparse.coo_matrix((x_d_values, (sparse_h.row, sparse_h.col)), shape=(m, n))
        x_d.append(np.array(np.abs(x_d_coo).sum(axis=0)).squeeze())
        x_d[d][x_d[d] == 0] = 1
        actual_sparse_h = sparse.coo_matrix((sparse_h.data * (x_argmax_values + x_argmin_values), (sparse_h.row, sparse_h.col)), shape=(m, n))
        gradient.append(np.subtract(x[:, d] - s[:, d], ((actual_sparse_h.T @ y[d]).T / x_d[d]).T))

    gradient = np.array(gradient).T
    y = np.array(y).T
    fx = sum([w * np.linalg.norm(x[j] - y[i], ord=np.inf)**2 for i, j, w in zip(sparse_h.row, sparse_h.col, sparse_h.data)]) / 2 - np.linalg.norm(x.T @ s, ord=2)
    return gradient, y, fx


def nonvectorized_infinity(x, s, sparse_h, rank, D):
    hypergraph = []
    he = sparse_h.col
    k = 0
    for r in rank:
        hypergraph.append([he[j] for j in range(k, k + int(r))])
        k += int(r)
    gradient = np.zeros(x.shape)
    degree = np.zeros(x.shape)
    y = np.zeros((len(rank), x.shape[-1]))
    fx = 0
    for i, e in enumerate(hypergraph):
        xe = x[e]
        y_max = xe.max(axis=0)
        y_min = xe.min(axis=0)
        y[i, :] = y_min + (y_max - y_min) / 2
        argmax = (xe == y_max)
        argmin = (xe == y_min)
        degree[e] += (argmax | argmin)
        gradient[e] += (xe - y[i, :]) * (argmax.astype(int) + argmin.astype(int))
        fx += np.linalg.norm(y[i, :] - y_min, ord=np.inf)
    degree[degree == 0] = 1
    gradient /= degree
    gradient -= (s.T / D).T
    # fx -= np.einsum('ij,ij->', x, s)
    return gradient, y, fx


diffusion_functions = OrderedDict([
    ('quadratic', quadratic),
    ('linear', linear),
    ('infinity', nonvectorized_infinity),
])
