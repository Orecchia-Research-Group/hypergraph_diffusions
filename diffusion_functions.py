'''
A set of diffusion functions over hypergraphs


'''

from datetime import datetime
from collections import OrderedDict
import numpy as np
from scipy import sparse


EPS = 1e-6


def quadratic(n, m, D, hypergraph, x0, s=None, h=1e-3, eps=EPS, verbose=0):
    '''
    Quadratic diffusion

    \\bar{\\delta_h(x)} = \\sum_{i \\in h} x_i - \\bar{x_h}

    The diffusion is

    \\partial \\bar{\\delta_h(x)} = \\sign(x_h  - \\bar{x_h})
    '''
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
    x_bar = [np.divide((sparse_h @ x[-1]).T, rank).T]
    fx = [sum([w * np.linalg.norm(x[-1][j] - x_bar[-1][i])**2 for i, j, w in zip(sparse_h.row, sparse_h.col, sparse_h.data)])]
    crit = 1
    t = 1
    t_start = datetime.now()
    print('{:>10s} {:>6s} {:>13s} {:>14s}'.format('Time (s)', '# Iter', '||dx||_D^2', 'F(x(t))'))
    while len(x) < 2 or crit > eps:
        t_now = datetime.now()
        print(f'\r{(t_now - t_start).total_seconds():10.3f} {t:6d} {crit:13.6f} {float(fx[-1]):14.6f}', end='')
        gradient = np.subtract(x[-1] - s, ((sparse_h.T @ x_bar[-1]).T / D).T)
        # print(gradient.max(), gradient.min())
        # print('x', x[-1].min(), x[-1].max())
        # print('x_bar', x_bar[-1].min(), x_bar[-1].max())
        # print('gradient', gradient.min(), gradient.max())
        # print(x_bar[-1][0], x[-1][hypergraph[0]], gradient[hypergraph[0]])
        # input()
        x.append(x[-1] - h * gradient)
        x_bar.append(np.divide((sparse_h @ x[-1]).T, rank).T)
        fx.append(sum([w * np.linalg.norm(x[-1][j] - x_bar[-1][i])**2 for i, j, w in zip(sparse_h.row, sparse_h.col, sparse_h.data)]))
        crit = np.linalg.norm((D * (x[-1] - x[-2]).T) @ (x[-1] - x[-2]))
        t += 1
    print()
    return np.array(x), np.array(x_bar), np.array(fx)


diffusion_functions = OrderedDict([
    ('quadratic', quadratic),
])
