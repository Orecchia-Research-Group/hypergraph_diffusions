"""Ikeda paper implementation and plots"""

import os
import argparse
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import reading
from diffusion_functions import diffusion_functions, diffusion, all_sweep_cuts, added_terms, compute_hypergraph_matrices, regularizers

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


def get_function_values(x, func, iterations, sparse_h, rank, W, node_weights, s, lamda, verbose=0):
    fx = np.zeros((x.shape[0], x.shape[-1]))
    if verbose > 0:
        print(f'{"Time (s)":8s} {"t":5s}')
    start_time = time.time()
    for t in range(fx.shape[0]):
        if verbose > 0:
            print(f'{time.time() - start_time:8.3f} {t:5d}', end='\r')
        _, _, fx[t] = func(x[t], sparse_h, rank, W, node_weights)
        fx[t] -= (x[t] * s).sum(axis=0)
        fx[t] += ((node_weights * x[t].T).T * x[t]).sum(axis=0) * lamda / 2
    if verbose > 0:
        print()
    return fx


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
    parser.add_argument('--lamda', '-l', help='Parameter used in personalized pagerank.', type=float, default=0)
    parser.add_argument('-e', '--eta', help='Exponential averaging parameter.', type=float, default=0.9)
    parser.add_argument('--regularizer', help='Preconditioner for hypergraph diffusion', choices=regularizers.keys(), default=tuple(regularizers.keys())[0])
    parser.add_argument('--save-folder', help='Folder to save pictures.', default=SAVE_FOLDER)
    parser.add_argument('--no-sweep', help='Disable doing sweep cuts.', action='store_true')
    parser.add_argument('--write-values', help='Save all values in a pickle file based on the arguments in the save folder.', action='store_true')
    parser.add_argument('-v', '--verbose', help='Verbose mode. Prints out useful information. Higher levels print more information.', action='count', default=0)
    args = parser.parse_args()
    return args


def main():
    """Driver"""
    args = parse_args()
    if args.verbose > 0:
        for name, value in vars(args).items():
            print(f'{name:10s} = {value}')
    graph_name = os.path.basename(os.path.splitext(args.hypergraph)[0])
    if args.write_values:
        pickle_filename = os.path.join(args.save_folder,
                                       f'Ikeda_{graph_name}_{args.function}_{args.regularizer}_{100 * args.lamda:.0f}.pickle')
        if os.path.isfile(pickle_filename):
            print("Pickle file exists. Exiting...")
            return
    if args.verbose > 0:
        print(f'Reading hypergraph from file {args.hypergraph}')
    func = diffusion_functions['infinity']
    n, m, node_weights, hypergraph, weights, center_id, hypergraph_node_weights = reading.read_hypergraph(args.hypergraph)
    if args.random_seed is None:
        args.random_seed = np.random.randint(1000000)
    np.random.seed(args.random_seed)
    vs = np.random.choice(np.arange(n), size=args.dimensions, replace=False)
    x0 = np.zeros((n, args.dimensions))
    s = np.zeros((n, args.dimensions))
    for d, v in enumerate(vs):
        s[v, d] = 1
    iteration_times, x, _, fx = diffusion(x0, n, m, node_weights, hypergraph, weights, s=s, lamda=args.lamda, center_id=center_id,
                         hypergraph_node_weights=hypergraph_node_weights, func=func,
                         h=args.step_size, T=args.iterations, regularizer=args.regularizer, verbose=args.verbose)
    if not args.no_sweep:
        value, volume, conductance = all_sweep_cuts(x, n, m, node_weights, hypergraph, args.verbose)
        results = conductance.min(axis=2)

        plot_surf(results)
        plt.savefig(os.path.join(args.save_folder, f'Ikeda_{graph_name}_{args.function}_{args.regularizer}_{100 * args.alpha:.0f}.png'), dpi=300)
        plt.show()

        plot_hist(results)
        plt.savefig(os.path.join(args.save_folder, f'Ikeda_{graph_name}_{args.function}_{args.regularizer}_{100 * args.alpha:.0f}_hist.png'), dpi=300)
        plt.show()

    W, sparse_h, rank = compute_hypergraph_matrices(n, m, hypergraph, weights)
    x_cs = np.cumsum(x, axis=0)
    if args.verbose > 0:
        print('Computing function values averaging over t')
    for t in range(x.shape[0]):
        x_cs[t] /= (t + 1)
    fx_cs = get_function_values(x_cs, func, args.iterations, sparse_h, rank, W, node_weights,
                                s, args.lamda, args.verbose)
    """
    final_x = np.zeros_like(x)
    if args.verbose > 0:
        print('Computing function values tail averaging over t')
    for t in range(x.shape[0]):
        final_x[t] = (x_cs[-1] * args.iterations - x_cs[t - 1] * t) / (args.iterations - t)
    final_fx = get_function_values(final_x, func, args.iterations, sparse_h, rank, W, node_weights,
                                   np.zeros_like(x0), x0, args.alpha, args.verbose)
    """
    exp_x = np.zeros_like(x)
    exp_x[0] = x0
    if args.verbose > 0:
        print('Computing function values with exponential weight averaging over t')
    for t in range(1, x.shape[0]):
        exp_x[t] = exp_x[t-1] * args.eta + x[t] * (1 - args.eta)
    exp_fx = get_function_values(exp_x, func, args.iterations, sparse_h, rank, W, node_weights,
                                 s, args.lamda, args.verbose)

    if args.verbose > 0:
        print('Min values')
        print(f'{"Last iterate":20s} = {fx.min():10.6f}')
        print(f'{"Averaging":20s} = {fx_cs.min():10.6f}')
        # print(f'{"Tail Averaging":20s} = {final_fx.min():10.6f}')
        print(f'{"Exponential Averaging":20s} = {exp_fx.min():10.6f}')

    # plt.figure()
    # plt.plot(fx_cs)
    # plt.show()

    plt.plot(np.min(fx, axis=1), label='Last iterate')
    plt.plot(np.min(fx_cs, axis=1), label='Average iterate')
    # plt.plot(np.min(final_fx, axis=1), label='Average tail iterate')
    plt.plot(np.min(exp_fx, axis=1), label='Exponential average iterate')
    plt.legend(loc='best')
    plt.xlabel('t')
    plt.ylabel('Q(x)')
    plt.savefig(os.path.join(args.save_folder, f'Ikeda_{graph_name}_{args.function}_{args.regularizer}_{100 * args.lamda:.0f}_value.png'), dpi=300)
    # plt.show()
    if args.write_values:
        pickle_filename = os.path.join(args.save_folder, f'Ikeda_{graph_name}_{args.function}_{args.regularizer}_{100 * args.lamda:.0f}.pickle')
        with open(pickle_filename, 'wb') as fp:
            pickle.dump({
                't': iteration_times,
                'x': x[-1],
                'x_cs': x_cs[-1],
                'exp_x': exp_x[-1],
                'fx': fx,
                'fx_cs': fx_cs,
                # 'final_fx': final_fx,
                'exp_fx': exp_fx,
            }, fp)


if __name__ == '__main__':
    main()
