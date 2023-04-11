"""Ikeda paper implementation and plots"""

import os
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import reading
from diffusion_functions import diffusion_functions, diffusion, all_sweep_cuts, added_terms, compute_hypergraph_matrices

SAVE_FOLDER = 'results'
STEP_SIZE = 1


def plot_surf(results):
    overall = np.zeros(results.shape)
    overall[0, 0] = results[0, 0]
    for t in range(results.shape[0]):
        for d in range(results.shape[1]):
            val = results[t, d]
            if t > 0:
                val = min(val, overall[t - 1, d])
            if d > 0:
                val = min(val, overall[t, d - 1])
            overall[t, d] = val

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.arange(overall.shape[1])
    Y = np.arange(overall.shape[0])
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, overall, cmap=cm.coolwarm, linewidth=0)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('# Restarts')
    ax.set_ylabel('T')
    ax.set_zlabel('Conductance')
    ax.view_init(azim=50)


def plot_hist(results):
    bins = 20
    result_hist = np.zeros((results.shape[0], bins))
    for t, rt in enumerate(results):
        result_hist[t], bin_edges = np.histogram(rt, bins=np.linspace(results.min(), results.max(), bins + 1))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = bin_edges[:-1]
    Y = np.arange(result_hist.shape[0])
    X, Y = np.meshgrid(X, Y)
    X, Y = X.ravel(), Y.ravel()
    bottom = np.zeros_like(X)
    depth = 1
    width = bin_edges[1] - bin_edges[0]
    ax.bar3d(X, Y, bottom, width, depth, result_hist.ravel(), shade=True)
    ax.view_init(azim=-40)
    ax.set_xlabel('Conductance')
    ax.set_ylabel('T')
    ax.set_zlabel('# of results')


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser(prog='Ikeda Practical Evaluation',
                                     description='Run practical experiments using the method described in Ikeda et al.',
                                     epilog='Konstantinos Ameranis, University of Chicago 2023')
    parser.add_argument('-g', '--hypergraph', help='Filename of hypergraph to use.', type=str, required=True)
    parser.add_argument('--step-size', help='Step size value.', type=float, default=STEP_SIZE)
    parser.add_argument('-f', '--function', help='Which diffusion function to use.', choices=diffusion_functions.keys(),
                        default=list(diffusion_functions.keys())[0])
    parser.add_argument('-r', '--random-seed', help='Random seed to use for initialization.', type=int, default=None)
    parser.add_argument('-d', '--dimensions', help='Number of embedding dimensions.', type=int, default=2)
    parser.add_argument('-T', '--iterations', help='Maximum iterations for diffusion.', type=int, default=20)
    parser.add_argument('--alpha', '-a', help='Parameter used in personalized pagerank.', type=float, default=0)
    parser.add_argument('--save-folder', help='Folder to save pictures.', default=SAVE_FOLDER)
    parser.add_argument('--no-sweep', help='Disable doing sweep cuts.', action='store_true')
    parser.add_argument('-v', '--verbose', help='Verbose mode. Prints out useful information. Higher levels print more information.', action='count', default=0)
    args = parser.parse_args()
    return args


def main():
    """Driver"""
    args = parse_args()
    graph_name = os.path.basename(os.path.splitext(args.hypergraph)[0])
    if args.verbose > 0:
        print(f'Reading hypergraph from file {args.hypergraph}')
    func = diffusion_functions['infinity']
    n, m, node_weights, hypergraph, weights, center_id, hypergraph_node_weights = reading.read_hypergraph(args.hypergraph)
    if args.random_seed is None:
        args.random_seed = np.random.randint(1000000)
    np.random.seed(args.random_seed)
    vs = np.random.choice(np.arange(n), size=args.dimensions, replace=False)
    x0 = np.zeros((n, args.dimensions))
    for d, v in enumerate(vs):
        x0[v, d] = 1
    x, _, fx = diffusion(x0, n, m, node_weights, hypergraph, weights, s=x0, alpha=args.alpha, center_id=center_id,
                         hypergraph_node_weights=hypergraph_node_weights, func=func,
                         h=args.step_size, T=args.iterations, verbose=args.verbose)
    if not args.no_sweep:
        value, volume, conductance = all_sweep_cuts(x, n, m, node_weights, hypergraph, args.verbose)
        results = conductance.min(axis=2)

        plot_surf(results)
        plt.savefig(os.path.join(args.save_folder, f'Ikeda_{graph_name}_{args.function}.png'), dpi=300)
        plt.show()

        plot_hist(results)
        plt.savefig(os.path.join(args.save_folder, f'Ikeda_{graph_name}_{args.function}_hist.png'), dpi=300)
        plt.show()

    W, sparse_h, rank = compute_hypergraph_matrices(n, m, hypergraph, weights)
    x_cs = np.cumsum(x, axis=0)
    fx_cs = np.zeros(fx.shape)
    if args.verbose > 0:
        print('Computing function values averaging over t')
        print(f'{"Time (s)":8s} {"t":>5s}')
    start_time = time.time()
    for t in range(args.iterations):
        if args.verbose > 0:
            print(f'{time.time() - start_time:8.3f} {t:5d}', end='\r')
        x_cs[t] /= (t + 1)
        _, _, fx_cs[t] = func(x_cs[t], sparse_h, rank, W, node_weights)
        _, fx_cs[t] = added_terms(x_cs[t], np.zeros_like(x0), fx_cs[t], node_weights, np.zeros_like(x0), x0, 2 * args.alpha / (1 + args.alpha))
    if args.verbose > 0:
        print()

    final_x = np.zeros_like(x)
    final_fx = np.zeros_like(fx)
    if args.verbose > 0:
        print('Computing function values averaging over t')
        print(f'{"Time (s)":8s} {"t":5s}')
    start_time = time.time()
    for t in range(args.iterations):
        if args.verbose > 0:
            print(f'{time.time() - start_time:8.3f} {t:5d}', end='\r')
        final_x[t] = (x_cs[-1] * args.iterations - x_cs[t - 1] * t) / (args.iterations - t)
        _, _, final_fx[t] = func(final_x[t], sparse_h, rank, W, node_weights)
        _, final_fx[t] = added_terms(final_x[t], np.zeros_like(x0), final_fx[t], node_weights,
                                                         np.zeros_like(x0), x0, 2 * args.alpha / (1 + args.alpha))
    if args.verbose > 0:
        print()
        print(f'Min values\n\n{"Last iterate":20s} = {fx.min() / x.shape[-1]:10.6f}\n{"Averaging":20s} = {fx_cs.min() / x.shape[-1]:10.6f}\n{"Tail Averaging":20s} = {final_fx.min() / x.shape[-1]:10.6f}')

    plt.plot(fx / x.shape[-1], label='Last iterate')
    plt.plot(fx_cs / x.shape[-1], label='Average iterate')
    plt.plot(final_fx / x.shape[-1], label='Average tail iterate')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.ylabel('Q(x)')
    plt.savefig(os.path.join(args.save_folder, f'Ikeda_{graph_name}_{args.function}_value.png'), dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
